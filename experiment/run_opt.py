import torch.nn as nn
from tern_svd import *
from collections import OrderedDict
from multiprocessing.pool import ThreadPool as Pool

from torchprofile import count_mul_and_add_for_first_input

DEV = torch.device('cuda:0')

def mul_device_add(A, B):
    if isinstance(B, torch.Tensor):
        B = B.to(A.device)
    return A.add(B)

def mul_device_iadd(A, B):
    if isinstance(B, torch.Tensor):
        B = B.to(A.device)
    return A.add_(B)

@contextmanager
def replace_add():
    with patch.object(torch.Tensor, "__iadd__", mul_device_iadd):
        with patch.object(torch.Tensor, "__add__", mul_device_add):
            yield

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    with replace_Linear_to_ternary_SVD_linear():
        model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

def bin_svd(model):
    if args.bin_svd_tol > 0:
        trans_fun = transform_policy(steps=20, tolerance=args.bin_svd_tol, verbose=True, cos_thresh=args.bin_svd_cos_theta)

        def run(M_list):
            for M in M_list[::-1]:
                dtype = M.weight.dtype
                M.float()
                M.weight_to_usv(trans_fun, None, prune_rate=float('Inf'))
                M.to(dtype)
                #M._parameters['weight'].data = M._parameters['weight'].to(dtype)
                M._buffers['u'] = M._buffers['u'].to(torch.int8)#.cpu()
                M._buffers['v'] = M._buffers['v'].to(torch.int8)#.cpu()
                del M.weight
                del M.rest_weight

        layer_list = {}

        @tern_svd_layer_patch
        def _trans(M):
            device = M.weight.device
            if device not in layer_list:
                layer_list[device] = list()
            layer_list[device].append(M)
        model.apply(_trans)

        with Pool() as p:
            tasks = [p.apply_async(run, (M_list,)) for d, M_list in layer_list.items()]
            for t in tasks:
                t.get()

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        if args.bin_svd_tol > 0:
            trans_fun = transform_policy(steps=20, tolerance=args.bin_svd_tol, verbose=True, cos_thresh=args.bin_svd_cos_theta)

            @tern_svd_layer_patch
            def _trans(M):
                if hasattr(M, 'weight'):
                    M.float()
                    M.weight_to_usv(trans_fun, None, prune_rate=float('Inf'))
                    del M.weight
                    del M.rest_weight
                    M.half()
            layer.apply(_trans)

        layer = count_mul_and_add_for_first_input(layer)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

@torch.no_grad()
def dist_opt_eval(model, testenc, dev):
    print('Evaluating ...')
    nsamples = testenc['input_ids'].numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    def dataset():
        for i in range(nsamples):
            def cut(t):
                return t[..., (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)

            batch = {key: cut(t) for key, t in testenc.items()}
            yield batch

    nlls = []
    for b in dataset():
        outs = model(**b)
        lm_logits = outs['logits'].to(dev)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = b['input_ids'][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


def opt_multigpu(model):
    gpus = [torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count())]

    i = 0
    def get_gpu():
        nonlocal i
        device = gpus[ i % torch.cuda.device_count()]
        i += 1
        return device

    class MoveModule(nn.Module):
        def __init__(self, module, device, recurse=True):
            super().__init__()
            device_past =  list({v.device for v in module.state_dict().values() if v.device != torch.device('cpu')})
            if len(device_past) > 0:
                assert len(device_past) == 1
                self.dev = device_past[0]
            else:
                self.dev = device
            if recurse:
                module.to(self.dev)
            else:
                modules = module._modules
                module._modules = OrderedDict()
                module.to(self.dev)
                module._modules = modules
            self.module = module

        def turn_device(self, p):
            if isinstance(p, torch.Tensor):
                return p.to(self.dev) if p.device != self.dev else p
            else:
                return p

        def forward(self, *inp, **kwargs):
            with replace_add():
                inp = [self.turn_device(p) for p in inp]
                kwargs = {n:self.turn_device(p) for n, p in kwargs.items()}
                tmp = self.module(*inp, **kwargs)
                return tmp

        def __iter__(self):
            return self.module.__iter__()

        def extra_repr(self) -> str:
            return 'device={}'.format(
               self.dev
            )

        def __getattr__(self, item):
            try:
                return super().__getattr__(item)
            except AttributeError:
                return getattr(self.module, item)

    block_layer_name = {'OPTDecoderLayer'}

    def apply(M):
        if type(M).__name__ in block_layer_name:
            M = MoveModule(M, get_gpu())
        else:
            M = MoveModule(M, get_gpu(), recurse=False)
            for name, mod in list(M.module._modules.items()):
                M.module._modules[name] = apply(mod)
        return M

    model = apply(model)
    return model

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        "--bin_svd_tol", type=float, default=0, help="tolerance of bin svd."
    )
    parser.add_argument(
        "--bin_svd_cos_theta", type=float, default=0.8386, help="cos theta of bin svd."
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )

    args = parser.parse_args()
    dataloader, testloader = get_loaders(
        'c4', seed=args.seed, model=args.model, seqlen=2048
    )
    print('c4')

    model = get_opt(args.model)
    model = opt_multigpu(model)
    bin_svd(model)
    model.eval()
    model = count_mul_and_add_for_first_input(model)
    dist_opt_eval(model, testloader, DEV)

    '''
    datasets = ['wikitext2', 'ptb', 'c4']
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        dist_opt_eval(model, testloader, DEV)
    '''