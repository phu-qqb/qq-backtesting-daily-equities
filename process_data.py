import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- Charger les fichiers historiques ---
# Format: security,datetime,close (sans header)
def load_multiseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=['security','datetime','close'],
        parse_dates=['datetime']
    )
    # Pivot en DataFrame datetime × security
    return df.pivot(index='datetime', columns='security', values='close').sort_index()

# Répertoire des fichiers
DATA_DIR = Path(r'D:/QQ/Intraday')

# Fichiers à charger
files = ['A.txt', 'H.txt', 'I.txt', 'A_EU.txt', 'H_EU.txt', 'I_EU.txt']
hist = {f: load_multiseries(DATA_DIR / f) for f in files}

# --- Supprimer le dernier jour complet de l'history pour certains fichiers ---
for key, df in hist.items():
    if key in ['A.txt','H.txt','A_EU.txt','H_EU.txt']:
        last_day = df.index.normalize().max()
        # Retirer toutes les lignes de ce jour
        mask = df.index.normalize() != last_day
        hist[key] = df.loc[mask]

# Extraire variables
df_A, df_H, df_I, df_A_EU, df_H_EU, df_I_EU = [hist[f] for f in files]

# --- Vérification flattening initial A vs H ---
rets_H = df_H.pct_change(fill_method=None)
rets_H.loc[rets_H.index.to_series().dt.normalize().diff()>pd.Timedelta(0),:] = 0.0
rets_A = df_A.pct_change(fill_method=None)
rets_A.loc[rets_A.index.to_series().dt.normalize().diff()>pd.Timedelta(0),:] = 0.0
rets_H.fillna(0, inplace=True)
rets_A.fillna(0, inplace=True)
if not np.allclose(rets_H.values, rets_A.values, atol=1e-7):
    print('Erreur initiale: returns H != returns A')
    sys.exit(1)
print('Vérification initiale OK')

# --- Charger et filtrer new_points.csv 13h-19h ---
new_df = pd.read_csv(DATA_DIR / 'new_points.csv', index_col=0, parse_dates=True)
new_df.columns = new_df.columns.astype(df_H.columns.dtype)
new_df = new_df[(new_df.index.hour>=13)&(new_df.index.hour<=19)]
# Exclure <= dernière date de A
last_A = df_A.index.max()
new_df = new_df[new_df.index>last_A]
print(f'Nouvelles barres: {len(new_df)}')

# --- Calcul returns nouveaux points ---
rets_new = new_df.pct_change(fill_method=None)
rets_new.iloc[0,:]=0.0
rets_new.loc[rets_new.index.to_series().dt.normalize().diff()>pd.Timedelta(0),:]=0.0
rets_new.fillna(0,inplace=True)

# --- Fusion returns (override) ---
rets_A = rets_A.drop(rets_new.index, errors='ignore')
rets_all = pd.concat([rets_A, rets_new]).sort_index()

# --- Reconstruction backward A_updated ---
tmp_last = df_A.iloc[-1].copy(); tmp_last.update(new_df.iloc[-1])
def reconstruct_series(r, last_p): return last_p*(1+r).cumprod()/((1+r).cumprod().iloc[-1])
flat = [reconstruct_series(rets_all[col], tmp_last[col]).rename(col) for col in rets_all]
A_updated = pd.concat(flat, axis=1).stack().rename('close').rename_axis(['datetime','security'])

# --- Mise à jour H_updated & I_updated ---
for label, df_orig in [('H',df_H),('I',df_I)]:
    filt = df_orig[(df_orig.index.hour>=13)&(df_orig.index.hour<=19)]
    concat = pd.concat([filt,new_df]).sort_index()
    concat = concat[~concat.index.duplicated(keep='last')]
    globals()[f'{label}_updated'] = concat.stack().rename('close').rename_axis(['datetime','security'])

# --- Sauvegarde A_updated & H_updated & I_updated ---
for label in ['A','H','I']:
    df_out = globals()[f'{label}_updated'].reset_index()[['security','datetime','close']]
    df_out.sort_values(['security','datetime'], inplace=True)
    df_out['datetime']=df_out['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    df_out.to_csv(DATA_DIR / f'{label}_updated.txt', header=False, index=False)
    print(f'{label}_updated.txt généré')

# --- Création D.txt à partir de A_updated timestamps ---
from pandas import Series, DatetimeIndex
ts = DatetimeIndex(hist['A.txt'].index).normalize() # use original A timestamps
# Actually use A_updated
ts = DatetimeIndex(A_updated.reset_index()['datetime'])
all_ts = ts.drop_duplicates().sort_values()
Series(all_ts.strftime('%Y-%m-%d %H:%M')).to_csv(DATA_DIR / 'D.txt', header=False, index=False)
print('D.txt généré')
