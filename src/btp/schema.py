"""Explicit source feature contract; diagnosis is metadata, never a predictor."""

DEMOGRAPHICS = ['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'PTRACCAT', 'PTETHCAT', 'PTMARRY']
CSF = ['ABETA_bl', 'TAU_bl', 'PTAU_bl']
COGNITION = [x + '_bl' for x in ['CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate',
    'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'FAQ', 'MOCA']]
MRI = [x + '_bl' for x in ['Hippocampus', 'Ventricles', 'WholeBrain', 'Entorhinal',
    'Fusiform', 'MidTemp', 'ICV']]
PET = ['FDG_bl', 'AV45_bl']
ECOG = [f'Ecog{who}{domain}_bl' for who in ['Pt', 'SP']
    for domain in ['Mem', 'Lang', 'Visspat', 'Plan', 'Organ', 'Divatt', 'Total']]
NEURO = ['TRAASCOR', 'TRABSCOR', 'DSPANFOR', 'DSPANBAC', 'CATANIMSC', 'CLOCKSCOR',
    'LIMMTOTAL', 'LDELTOTAL', 'BNTTOTAL', 'DIGITSCOR']
HISTORY = ['MH4CARD', 'MH9ENDO', 'MH16SMOK', 'MHPSYCH', 'MH3HEAD', 'MH14ALCH', 'MH2NEURL']
A_FEATURES = DEMOGRAPHICS + CSF + COGNITION + MRI + PET
B_FEATURES = ECOG + NEURO + HISTORY
FEATURES = A_FEATURES + B_FEATURES
CATEGORICAL = ['PTGENDER', 'PTRACCAT', 'PTETHCAT', 'PTMARRY']
PANELS = {
    'clinical': DEMOGRAPHICS[:4] + COGNITION,
    'core': DEMOGRAPHICS[:4] + CSF + COGNITION + MRI + ['FDG_bl'] + HISTORY[:3],
    'expanded': FEATURES,
}


def registry():
    rows = []
    for col in FEATURES:
        source = 'NEUROBAT' if col in NEURO else 'MEDHIST' if col in HISTORY else 'ADNIMERGE'
        temporal = ('baseline or screening within 90 days, no later than index' if col in NEURO
            else 'aggregate all records on/before index' if col in HISTORY
            else 'original baseline row; baseline measurement' if col.endswith('_bl')
            else 'original baseline row')
        rows.append(dict(feature=col, source=source, temporal_rule=temporal,
            type='category' if col in CATEGORICAL else 'numeric',
            tier=1 if col in PANELS['core'] else 2,
            units='source-table units; no unit conversion',
            missing_rule='field-specific codes; unknown history remains missing'))
    return rows
