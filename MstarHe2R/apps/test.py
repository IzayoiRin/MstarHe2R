def main(*args, **kwargs):
    print('*'*10 + "Successful Testing" + '*'*10)
    if args:
        print('*'*10 + "Args: %s" % args + '*'*10)
    if kwargs:
        print('*'*10 + "Kwargs: %s" % kwargs + '*'*10)
