from .utils import math

__all__ = ['handlers']


def addmm(node):
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    ops = n * m * p
    return ops, ops

def addmv(node):
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    n, m = node.inputs[1].shape
    ops = n * m
    return ops, ops


def bmm(node):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    ops = b * n * m * p
    return ops, ops


def baddbmm(node):
    # [b, n, p] = aten::baddbmm([b, n, p], [b, n, m], [b, m, p])
    b, n, p = node.inputs[0].shape
    b, n1, m = node.inputs[1].shape
    b, m1, p1 = node.inputs[2].shape
    assert n == n1 and m == m1 and p == p1
    ops = b * n * m * p
    return ops, ops


def matmul(node):
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = aten::matmul([n], [n])
        n = node.inputs[0].shape[0]
        ops = n
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = aten::matmul([n], [n, m])
        n, m = node.inputs[1].shape
        ops = n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = aten::matmul([n, m], [m])
        n, m = node.inputs[0].shape
        ops = n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = aten::matmul([n, m], [m, p])
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        ops = n * m * p
    elif node.inputs[0].ndim == 1:
        # [..., m] = aten::matmul([n], [..., n, m])
        *b, n, m = node.inputs[1].shape
        ops = math.prod(b) * n * m
    elif node.inputs[1].ndim == 1:
        # [..., n] = aten::matmul([..., n, m], [m])
        *b, n, m = node.inputs[0].shape
        ops = math.prod(b) * n * m
    else:
        # [..., n, p] = aten::matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        ops =  math.prod(b) * n * m * p
    if 'tern_svd_' in node.scope and node.operator == 'aten::linear':
        return ops, ops, node.inputs[1]
    else:
        return ops, ops


def elementswise_mul(node):
    os = node.outputs[0].shape
    return math.prod(os), 0

def elementswise_add(node):
    os = node.outputs[0].shape
    return 0, math.prod(os)


def convolution(node):
    if node.outputs[0].shape[1] == node.inputs[1].shape[0]:
        oc, ic, *ks = node.inputs[1].shape
    else:
        ic, oc, *ks = node.inputs[1].shape
    os = node.outputs[0].shape
    ops = math.prod(os) * ic * math.prod(ks)

    if 'tern_svd_' in node.scope:
        return ops, ops, node.inputs[1]
    else:
        return ops, ops


def norm(node):
    if node.operator in ['aten::batch_norm', 'aten::instance_norm']:
        affine = node.inputs[1].shape is not None
    elif node.operator in ['aten::layer_norm', 'aten::group_norm']:
        affine = node.inputs[2].shape is not None
    else:
        raise ValueError(node.operator)

    os = node.outputs[0].shape
    ops = math.prod(os) if affine else 0
    si = node.inputs[0].shape
    return ops, math.prod(si)

def softmax(node):
    si = node.inputs[0].shape
    ops = math.prod(si)
    return ops, ops


def avg_pool_or_mean(node):
    os = node.outputs[0].shape
    si = node.inputs[0].shape
    return math.prod(os), math.prod(si)

def sum(node):
    si = node.inputs[0].shape
    return 0, math.prod(si)

def upsample_bilinear2d(node):
    os = node.outputs[0].shape
    return math.prod(os) * 4


handlers = (
    ('aten::addmm', addmm),
    ('aten::addmv', addmv),
    ('aten::bmm', bmm),
    ('aten::baddbmm', baddbmm),
    (('aten::linear', 'aten::matmul'), matmul),
    (('aten::mul', 'aten::mul_', 'aten::gelu', 'aten::leaky_relu', 'aten::sqrt', 'aten::pow',
      'aten::div', 'aten::div_', 'aten::hardtanh_', 'aten::hardtanh', 'aten::sigmoid', 'aten::tanh'), elementswise_mul),
    (('aten::add', 'aten::add_', 'aten::sub', 'aten::sub_', 'aten::rsub'), elementswise_add),
    (('aten::sum', 'aten::cumsum'), sum),
    ('aten::_convolution', convolution),
    (('aten::batch_norm', 'aten::instance_norm', 'aten::layer_norm',
      'aten::group_norm'), norm),
    (('aten::adaptive_avg_pool1d', 'aten::adaptive_avg_pool2d',
      'aten::adaptive_avg_pool3d', 'aten::avg_pool1d', 'aten::avg_pool2d',
      'aten::avg_pool3d', 'aten::mean'), avg_pool_or_mean),
    ('aten::upsample_bilinear2d', upsample_bilinear2d),
    ('aten::softmax', softmax),
    (('aten::adaptive_max_pool1d', 'aten::adaptive_max_pool2d',
      'aten::adaptive_max_pool3d',
      'aten::alpha_dropout', 'aten::cat', 'aten::chunk', 'aten::clamp',
      'aten::clone', 'aten::constant_pad_nd', 'aten::contiguous',
      'aten::flatten_dense_tensors', 'aten::unflatten_dense_tensors', 'aten::copy_',
      'aten::detach', 'aten::dropout',
      'aten::dropout_', 'aten::embedding', 'aten::eq', 'aten::feature_dropout',
      'aten::flatten', 'aten::floor', 'aten::floor_divide', 'aten::gt',
      'aten::index', 'aten::int',
      'aten::lt', 'aten::max_pool1d', 'aten::max_pool1d_with_indices', 'aten::max', 'aten::reshape',
      'aten::max_pool2d', 'aten::max_pool2d_with_indices', 'aten::max_pool3d',
      'aten::max_pool3d_with_indices', 'aten::max_unpool1d',
      'aten::max_unpool2d', 'aten::max_unpool3d', 'aten::ne',
      'aten::reflection_pad1d', 'aten::reflection_pad2d',
      'aten::reflection_pad3d', 'aten::relu', 'aten::relu_',
      'aten::replication_pad1d', 'aten::replication_pad2d',
      'aten::replication_pad3d', 'aten::select',
      'aten::size', 'aten::slice',
      'aten::squeeze', 'aten::stack', 'aten::t',
      'aten::threshold', 'aten::to', 'aten::transpose',
      'aten::unsqueeze', 'aten::permute',
      'aten::upsample_nearest2d', 'aten::view', 'aten::zeros',
      'aten::scalarimplicit', "aten::full", "aten::arange", "aten::masked_fill_",
      "aten::expand", "aten::masked_fill", "aten::type_as",
      'prim::constant', 'prim::listconstruct', 'prim::listunpack',
      'profiler::_record_function_exit', 'profiler::_record_function_enter',
      'aten::im2col', 'aten::abs', 'aten::neg',
      "prim::tupleunpack", "prim::numtotensor",
      'prim::numtotensor', 'prim::tupleconstruct'), None),
)
