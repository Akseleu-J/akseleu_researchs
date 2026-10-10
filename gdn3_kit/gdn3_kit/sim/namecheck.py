import ast, sys, builtins
def check(path):
    src = open(path).read(); tree = ast.parse(src)
    defined = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: defined.add((a.asname or a.name).split('.')[0])
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            defined.add(n.name)
            if not isinstance(n, ast.ClassDef):
                for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs: defined.add(a.arg)
                if n.args.vararg: defined.add(n.args.vararg.arg)
                if n.args.kwarg: defined.add(n.args.kwarg.arg)
        elif isinstance(n, ast.Lambda):
            for a in n.args.args: defined.add(a.arg)
            if n.args.vararg: defined.add(n.args.vararg.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): defined.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name: defined.add(n.name)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name): defined.add(t.id)
    bad = sorted({(n.id, n.lineno) for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined})
    return bad
rc = 0
for p in sys.argv[1:]:
    b = check(p)
    if b: rc = 1; print(p, b)
print("namecheck done", "ISSUES" if rc else "clean")
