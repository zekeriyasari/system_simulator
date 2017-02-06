from inspect import Parameter, Signature, functools
import types


def dump(obj):
    tmpstr = ''
    for attr in dir(obj):
        tmpstr += "{} : {} \n".format(attr, getattr(obj, attr))
    return tmpstr


def make_signature(names):
    # Creates a parameter signature
    return Signature(
        Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in names)


def create_property(name, expected_type=None, check_func=None,
                    get_before_callback=None,
                    get_after_callback=None,
                    set_before_callback=None,
                    set_after_callback=None,
                    del_before_callback=None,
                    del_after_callback=None,
                    doc=None):
    """Create properties from class _fields dictionary of the Structure class
    Created property checks if passed value is of type expected_type,
    and runs check_func if the value is acceptable. default check_func name is
    name_check where name is the property.
    If is callable, set_before_callback is called before setting the property value
    If is callable, set_after_callback is called after setting property value
    If is callable, del_cllback is called when property is deleted.
    """
    storage_name = '_' + name

    @property
    def new_prop(self):
        if callable(get_before_callback):
            get_before_callback(self)
        result = getattr(self, storage_name)
        if callable(get_after_callback):
            get_after_callback(self, result)
        return result

    @new_prop.setter
    def new_prop(self, value):
        if expected_type is not None:
            if (isinstance(expected_type, tuple) and types.FunctionType in expected_type) \
                    or isinstance(expected_type, types.FunctionType):
                if not (check_functional_parameter(value) or check_lambda_parameter(value) or callable(value)):
                    raise TypeError('{} can not be set to {}.'.format(self.__class__.__name__ + '.' + name, value))
            elif not isinstance(value, expected_type):
                raise TypeError('{} must be a {}.'.format(name, expected_type))

        if callable(check_func):
            if not check_func(value):
                raise ValueError('{} can not be set to {}.'.format(self.__class__.__name__ + '.' + name, value))

        if callable(set_before_callback):
            set_before_callback(self, value)

        if check_functional_parameter(value):  # Fix the function if functional value is passed
            callback, callback_args, callback_kwargs = value
            setattr(self, storage_name, functools.partial(callback, *callback_args, **callback_kwargs))
        else:
            setattr(self, storage_name, value)

        if callable(set_after_callback):
            set_after_callback(self, value)

    @new_prop.deleter
    def new_prop(self):
        if callable(del_before_callback):
            del_before_callback(self)
        delattr(self, storage_name)
        if callable(del_after_callback):
            del_after_callback(self)

    new_prop.__doc__ = doc
    return new_prop


def check_functional_parameter(value):
    result = False
    if isinstance(value, tuple) and len(value) in (3, 4):
        result = callable(value[0]) and isinstance(value[1], tuple) and isinstance(value[2], dict)
    return result


def check_lambda_parameter(value):
    return isinstance(value, types.LambdaType) and value.__name__ == "<lambda>"


def check_parameter_with_type(value):
    result = False
    if isinstance(value, tuple) and len(value) in (2, 3):
        if isinstance(value[1], tuple):
            for type_ in value[1]:
                result |= isinstance(type_, type) and isinstance(value[0], type_)
        else:
            result |= isinstance(value[1], type) and isinstance(value[0], value[1])
    return result


class StructureMeta(type):
    def __new__(mcs, clsname, bases, clsdict):
        clsobj = super().__new__(mcs, clsname, bases, clsdict)
        fields = []
        for name, val in clsobj._parameters.items():
            if check_functional_parameter(val):
                property_default_value = functools.partial(val[0], *val[1], **val[2])
                property_type = (type(val[0]), tuple)
                if len(val) == 4:
                    property_doc = str(val[3])
                else:
                    property_doc = ''
            elif check_parameter_with_type(val):
                property_default_value, property_type = val[0:2]
                if len(val) == 3:
                    property_doc = val[2]
                else:
                    property_doc = ''
            else:
                property_default_value = val
                property_type = None
                property_doc = str(type(val))

            if hasattr(clsobj, name + '_check'):
                checkfunc = getattr(clsobj, name + '_check')
            else:
                checkfunc = None

            if hasattr(clsobj, name + '_get_before'):
                getfuncbefore = getattr(clsobj, name + '_get_before')
            else:
                getfuncbefore = None

            if hasattr(clsobj, name + '_get_after'):
                getfuncafter = getattr(clsobj, name + '_get_after')
            else:
                getfuncafter = None

            if hasattr(clsobj, name + '_set_before'):
                setfuncbefore = getattr(clsobj, name + '_set_before')
            else:
                setfuncbefore = None

            if hasattr(clsobj, name + '_set_after'):
                setfuncafter = getattr(clsobj, name + '_set_after')
            else:
                setfuncafter = None

            if hasattr(clsobj, name + '_del_before'):
                delfuncbefore = getattr(clsobj, name + '_del_before')
            else:
                delfuncbefore = None

            if hasattr(clsobj, name + '_del_after'):
                delfuncafter = getattr(clsobj, name + '_del_after')
            else:
                delfuncafter = None

            setattr(clsobj, name, create_property(name, expected_type=property_type,
                                                  check_func=checkfunc,
                                                  get_before_callback=getfuncbefore,
                                                  get_after_callback=getfuncafter,
                                                  set_before_callback=setfuncbefore,
                                                  set_after_callback=setfuncafter,
                                                  del_before_callback=delfuncbefore,
                                                  del_after_callback=delfuncafter,
                                                  doc=property_doc))
            setattr(clsobj, '_' + name, property_default_value)
            fields.append(name)

        sig = make_signature(fields)
        setattr(clsobj, '__signature__', sig)
        # delattr(clsobj, '_parameters')
        clsobj.count = 0
        return clsobj


class Structure(metaclass=StructureMeta):
    _parameters = {}
    _label = ''

    def __init__(self, *args, **kwargs):
        # init parameters from **kwargs
        # sig = self.__signature__
        for name, val in kwargs.items():
            if hasattr(self, name) and val is not None:
                if check_functional_parameter(val):
                    setattr(self, '_' + name, functools.partial(val[0], *val[1], **val[2]))
                else:
                    setattr(self, '_' + name, val)
        self.label = kwargs.get('label', '<{} {:x}>'.format(self.__class__.__qualname__, id(self)))
        self.__class__.count += 1
        Structure.count += 1

    def __del__(self):
        self.__class__.count -= 1
        Structure.count -= 1

    @property
    def label(self):
        """Object Label"""
        return self._label

    @label.setter
    def label(self, val):
        self._label = str(val)

    def clean_kwargs(self, **kwargs):
        for key, value in self._parameters.items():
            kwargs.pop(key, None)
        return kwargs
