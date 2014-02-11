import pymatlab

class diva_synth:
    def __init__(self, diva_path):
        self.session = pymatlab.session_factory()
        self.session.run('path(' + diva_path + ', path)')

    def execute(self, art):
        self.session.putvalue('art', art)
        self.session.run('[aud, som, outline] = diva_synth(art, \'audsom\')')
        return self.session.getvalue('aud')
    
    def stop(self):
        del self.session
