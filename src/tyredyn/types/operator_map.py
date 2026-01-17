import operator

# map symbolic operator to actual operator
_OPS = {
    "<":  operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">":  operator.gt,
}