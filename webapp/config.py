"""Flask configuration."""


class Config(object):
    LOCATION_CODES = {
        'Chicago': 'KMDW',
        'Cedar_Rapids': 'KCID',
        'Des_Moines': 'KDSM',
        'Madison': 'KMSN',
        'Quincy': 'KUIN',
        'St_Louis': 'KSTL',
        'Rochester': 'KRST',
        'Green_Bay': 'KGRB',
        'Lansing': 'KLAN',
        'Indianapolis': 'KIND',
        'Columbus': 'KCMH',
        'Toledo': 'KTOL'
    }
    TARGET_LOCATION = 'Chicago'
    TARGET_TIMEZONE = 'America/Chicago'
    MAX_LOOK_BACK_HOURS = 24
