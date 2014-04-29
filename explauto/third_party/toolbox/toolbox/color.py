
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


def enable():
    global bold, end
    global   black,   red,   green,   yellow,   blue,   purple,   cyan,   grey
    global  bblack,  bred,  bgreen,  byellow,  bblue,  bpurple,  bcyan,  bgrey
    global  iblack,  ired,  igreen,  iyellow,  iblue,  ipurple,  icyan,  igrey
    global biblack, bired, bigreen, biyellow, biblue, bipurple, bicyan, bigrey

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

def disable():
    global bold, end
    global   black,   red,   green,   yellow,   blue,   purple,   cyan,   grey
    global  bblack,  bred,  bgreen,  byellow,  bblue,  bpurple,  bcyan,  bgrey
    global  iblack,  ired,  igreen,  iyellow,  iblue,  ipurple,  icyan,  igrey
    global biblack, bired, bigreen, biyellow, biblue, bipurple, bicyan, bigrey
    bold    = ''
    end     = ''

    black   = ''
    red     = ''
    green   = ''
    yellow  = ''
    blue    = ''
    purple  = ''
    cyan    = ''
    grey    = ''

    # Bold
    bblack  = ''
    bred    = ''
    bgreen  = ''
    byellow = ''
    bylue   = ''
    bpurple = ''
    bcyan   = ''
    bgrey   = ''

    # High Intensty
    iblack  = ''
    ired    = ''
    igreen  = ''
    iyellow = ''
    iblue   = ''
    ipurple = ''
    icyan   = ''
    igrey   = ''

    # Bold High Intensty
    biblack = ''
    bired   = ''
    bigreen = ''
    biyellow= ''
    biblue  = ''
    bipurple= ''
    bicyan  = ''
    bigrey  = ''

enable()
