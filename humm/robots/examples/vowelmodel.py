import robots

vm = robots.VowelModel()

order = (0.5, 0.5, 0.5)
result = tuple(vm.execute_order(order))
print "order: %s \tresult: %s" % (order, result)
