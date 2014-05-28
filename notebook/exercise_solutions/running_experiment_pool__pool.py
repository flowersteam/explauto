"""
We will now run each experiment 10 times.
So this will take quite some time!

Then, we show an exemple of how to plot the mean learning curve of each experiment.

"""

xps.run(repeat=10)

for log in xps.logs:
    avg_err = mean([mean(array(l.eval_errors), axis=1) for l in log], axis=0)
    std_err = std([mean(array(l.eval_errors), axis=1) for l in log], axis=0)

    errorbar(log[0].eval_at, avg_err, std_err)
    
legend(('motor', 'goal'))
