from rdkit import RDLogger

def bootstrap():
    ########## DISABLE RDKIT LOGGING
    RDLogger.DisableLog('rdApp.*') 