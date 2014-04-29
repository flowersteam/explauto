from __future__ import print_function, division
import datetime

bold    = '\033[1m'
end     = '\033[0m'

black   = '\033[0;30m'
red     = '\033[0;31m'
green   = '\033[0;32m'
yellow  = '\033[0;33m'
blue    = '\033[0;34m'
purple  = '\033[0;35m'
cyan    = '\033[0;36m'
grey    = '\033[0;37m'

# Bold
bblack  = '\033[1;30m'
bred    = '\033[1;31m'
bgreen  = '\033[1;32m'
byellow = '\033[1;33m'
bylue   = '\033[1;34m'
bpurple = '\033[1;35m'
bcyan   = '\033[1;36m'
bgrey   = '\033[1;37m'

# High Intensty
iblack  = '\033[0;90m'
ired    = '\033[0;91m'
igreen  = '\033[0;92m'
iyellow = '\033[0;93m'
iblue   = '\033[0;94m'
ipurple = '\033[0;95m'
icyan   = '\033[0;96m'
igrey   = '\033[0;97m'

# Bold High Intensty
biblack = '\033[1;90m'
bired   = '\033[1;91m'
bigreen = '\033[1;92m'
biyellow= '\033[1;93m'
biblue  = '\033[1;94m'
bipurple= '\033[1;95m'
bicyan  = '\033[1;96m'
bigrey  = '\033[1;97m'

import sys, types, time

prg_startcount  = 0
prg_startdate   = 0
prg_lastdisplay = None

def repr_vector(v, fmt = ' 5.2f'):
    fmt = '{:%s}' % fmt
    body = ', '.join(fmt.format(v_i) for v_i in v)
    if type(v) == tuple:
        return '('+body+')'
    else:
        return '['+body+']'

ppv = repr_vector

def repr_timedelta(td):
    """Represent a timedelta object in a human-readable way
    Maximum length for period < 1week = 11.
    """
    r = ""
    if td.days > 0:
        r += ("%id" % (td.days,))
    h = td.seconds // 3600
    t = td.seconds - 3600*h
    m = t // 60
    s = t - 60 * m
    if h > 0:
        r += ("%ih" % (h,))
    if m > 0:
        r += ("%im" % (m,))
    r += ("%is" % (s,))
    return r

def print_progress(done, todo, prefix = "", suffix = end, quiet = False, eta = None, freq = 0.5):
    """
    Print the progress bar with custom prefix and suffix.

    This function will print the progress, and return to the beginning of the line without
    creating a new line. This allow multiple call to this function to 'update the display'.
    Note that when done = todo, the function will print a new line at the end of the
    progress bar.
    This function has lot of options, to allow a one liner in your code in most cases.
    @param done    number describing how much progress as been done.
                   in a 'for i in range(n): ...' you usually want to call print_progress(i+1, n, ...)
    @param todo    total of what is to do. If done == todo, there is nothing left to do.
    @param prefix  convenience : what is printed before the progress bar. (ex: color code)
    @param suffix  convenience : what is printed after the progress bar. By default, return to white color.
                   It also adds 10 whitespace at the end to 'erase' longer preceding lines.
    @param quiet   convenience : variable to turn the printing on/off.
    @param eta     display ETA : estimated time of arrival (completion).
                   the paramter is a number indicating the done value to count as a start date (usually 1).
                   If in a subsequent call, done < eta, eta will be displayed as N/A.
                   This allow to correctly diplay ETA for resuming of partial progress.
    @param freq    How often should the progress be updated.
                   If a integer is provided, the progress bar is updated when done modulo freq = 0.
                   If a float is provided, the progress bar is updated every freq seconds.
    """
    global prg_startcount, prg_startdate, prg_lastdisplay
    if not quiet:
        if done != todo:
            if type(freq) == types.FloatType:
                now = time.time()
                if prg_lastdisplay is None:
                    prg_lastdisplay = now
                elif now - prg_lastdisplay < freq:
                    return
                else:
                    prg_lastdisplay = now
            if type(freq) == types.IntType:
                if done % freq != 0:
                    return
        if eta is None:
            print(prefix, progressBar(done, todo), sep = '', end = suffix+'\r\033[1K')
        else:
            if done == eta:
                prg_startcount = eta
                prg_startdate  = datetime.datetime.now()
            if done <= eta:
                print(prefix, progressBar(done, todo), sep = '', end = ' ETA: N\A '+suffix+'\r\033[1K')
            else:
                eta_repr = repr_timedelta((todo-done)*(datetime.datetime.now() - prg_startdate)//done)
                print(prefix, progressBar(done, todo), sep = '', end = ' ETA: %s %s\r\033[1K' %  (eta_repr, suffix))
        if done == todo:
            print()
        sys.stdout.flush()

def progressBar(done, todo, count = True, size = 50):
    cutoff  = int(size*done/todo)
    percent = int( 100*done/todo)
    if count:
        return '[' + ('#'*cutoff) + ('.'*(size-cutoff)) + ('% 4i%%] %5i/%5i' % (percent, done, todo))
    else:
        return '[' + ('#'*cutoff) + ('.'*(size-cutoff)) + ('% 4i%%]' % percent)
