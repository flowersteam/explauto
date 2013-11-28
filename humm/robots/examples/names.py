import robots

print robots.robotlist()

for name in robots.robotlist():
    try:
        realname = robots.robot(name).__repr__()
        print "%s instanciated." % (realname)
        assert realname.startswith(name), "%s %s" % (realname, name)
    except Exception:
        pass