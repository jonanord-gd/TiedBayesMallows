import rdata
import pandas as pd
import numpy as np

parsed = rdata.parser.parse_file(
    r'pan_cancer\Ciriello2013_2017-08_p1247_random.Rdata'
)
converted = rdata.conversion.convert(parsed)

out_dir = r'pan_cancer'

# R: ranking matrix (2617 x 1247)
R = converted['R']
pd.DataFrame(R.values, index=R.coords['dim_0'].values, columns=R.coords['dim_1'].values).to_csv(
    f'{out_dir}\\rankings.csv'
)
print("Saved rankings.csv")

# cancertype.subset: cancer type labels (2617,)
cancer_labels = converted['cancertype.subset']
pd.Series(cancer_labels, name='cancertype').to_csv(
    f'{out_dir}\\cancertype.csv', index=True
)
print("Saved cancertype.csv")

# gep.final: gene expression values (2617 x 1247)
gep = converted['gep.final']
pd.DataFrame(gep.values, index=gep.coords['dim_0'].values, columns=gep.coords['dim_1'].values).to_csv(
    f'{out_dir}\\gene_expression.csv'
)
print("Saved gene_expression.csv")

print("Done.")
