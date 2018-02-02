"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
function operation(target, name, descriptor) {
    var fn = descriptor.value;
    descriptor.value = function () {
        var args = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            args[_i] = arguments[_i];
        }
        return environment_1.ENV.math.scope(name, function () { return fn.apply(void 0, args); });
    };
    return descriptor;
}
exports.operation = operation;
//# sourceMappingURL=decorators.js.map