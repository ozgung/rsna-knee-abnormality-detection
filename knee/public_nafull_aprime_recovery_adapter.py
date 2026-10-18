"""Call the fixed A‴ recovery from the archived WS2 arm order."""
from knee import public_nafull_aprime as A

ARMS = ('old320', 'r7', 'crop224')
RECIPE = A.VERSION
WEIGHTS = A.WEIGHTS


def recover(old320, r7, crop224, old_missing, r7_missing, crop224_missing, *, fatal_errors=()):
    result=A.recover_ws2_order(old320, r7, crop224, old_missing, r7_missing, crop224_missing,
                               fatal_errors=fatal_errors)
    # The archived WS2 runner omits this diagnostic key when it writes JSON;
    # retain its schema while exposing the same raw float32 CSV roundtrip.
    result['source224_csv_values']=result['r224_csv_values']
    result.pop('r224_csv_values')
    return result
