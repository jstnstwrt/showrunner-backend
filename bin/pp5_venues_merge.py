#!/usr/bin/env python
"""
PP5 - Venues Master Merge
Merges venue data from all sources into the master venues table.
Runs on Zyte ScrapyCloud platform.
"""

import io
import json
import logging
import os
from datetime import date

import boto3
import emoji
import html
import numpy as np
import pandas as pd
import Levenshtein
import unidecode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_newest_file(s3, bucket, prefix, typ='csv'):
    results = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = sorted([obj['Key'] for obj in results.get('Contents', [])])
    latest = files[-1]
    logging.info("Loading: s3://%s/%s", bucket, latest)
    obj = s3.get_object(Bucket=bucket, Key=latest)
    if typ == 'csv':
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    elif typ == 'jsonlines':
        return pd.read_json(io.BytesIO(obj['Body'].read()), lines=True)


def write_file(s3, df, bucket, prefix, name):
    timestamp = str(date.today()).replace('-', '')
    key = f"{prefix}{name}_{timestamp}.csv"
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    logging.info("Wrote s3://%s/%s", bucket, key)


def update_df(df, df2, col, flag=1):
    df = df.drop_duplicates(subset=[col])
    df2 = df2.drop_duplicates(subset=[col])
    df = df.set_index(col)
    df2 = df2.set_index(col)
    df.update(df2, overwrite=flag)
    df = df.reset_index()
    return df


def pairwise_dist(df, df2, id1, id2, cols):
    return_df = []
    for col in cols:
        dt = df[[id1, col]]
        dt['key'] = 1
        dt2 = df2[[id2, col]]
        dt2['key'] = 1
        dt = dt2.merge(dt, on='key', suffixes=('', '2'))
        if dt[col].dtype == 'O':
            dt['dist'] = dt.apply(lambda x: Levenshtein.distance(x[col], x[col+'2']), axis=1)
            dt['score'] = dt.groupby(id2).dist.rank()
            dt = dt[(dt['score'] <= 3) | (dt['dist'] == 0)]
            dt['ratio'] = dt.dist / dt[col+'2'].str.len()
            dt = dt[dt.ratio < 0.5]
        else:
            dt['dist'] = (dt[col] - dt[col+'2']).abs()
            dt['score'] = dt.groupby(id2).dist.rank()
            dt = dt[dt['score'] <= 3]
        dt = dt.rename(columns={'dist': col+'_dist'})
        return_df.append(dt[[id2, id1, col+'_dist']])
    result = return_df[0]
    for r in return_df[1:]:
        result = result.merge(r, on=[id2, id1])
    return result


def match_df(df, df2, col):
    test_df = pairwise_dist(df, df2, 'venue_id', col, ['clean_name', 'lat', 'long'])
    test_df['dist'] = test_df['lat_dist'] + test_df['long_dist']
    test_df = test_df.sort_values([col, 'clean_name_dist', 'dist']).groupby(col).head(1)
    test_df = test_df.sort_values(['venue_id', 'clean_name_dist', 'dist']).groupby('venue_id').head(1)
    return test_df[['venue_id', col]]


def make_update(df, df2, col, flag=1):
    df = update_df(df, df2, col, flag)
    df2 = df2[~df2[col].isin(df[col])]
    if len(df2) == 0:
        return df
    dfm = match_df(df, df2, col)
    df = update_df(df, dfm, 'venue_id', 0)
    df = update_df(df, df2, col, flag)
    df2 = df2[~df2[col].isin(df[col])]
    if len(df2) == 0:
        return df
    df2 = df2.reset_index(drop=True).reset_index()
    df2 = df2.rename(columns={'index': 'venue_id'})
    df2.venue_id += df.venue_id.max() + 1
    df = pd.concat([df, df2])
    return df


def main():
    logging.info("=== PP5 Venues Merge ===")

    # Extract credentials from SHUB_SETTINGS and initialize S3 client
    settings = json.loads(os.environ["SHUB_SETTINGS"])["project_settings"]
    AWS_ACCESS_KEY_ID = settings["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = settings["AWS_SECRET_ACCESS_KEY"]

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    BUCKET = 'euclidsfund-data-pipeline'

    # --- Load prior merged venues and org table ---
    df = get_newest_file(s3, BUCKET, 'showrunner/merge/venues')
    df['venue_id'] = pd.to_numeric(df['venue_id'], errors='coerce')
    dfo = get_newest_file(s3, BUCKET, 'showrunner/merge/org')
    dfo['org_id'] = pd.to_numeric(dfo['org_id'], errors='coerce')
    # Preserve existing venue→org mappings before stripping org_id
    dfvo = df[['venue_id', 'org_id']].copy()
    del df['org_id']
    # Strip fair venue IDs (venue_id >= 1000000) from prior run
    df = df[df.venue_id < 1000000]
    logging.info("Loaded prior venues: %s", len(df))

    # --- Load and prepare seesaw venues ---
    dfss = get_newest_file(s3, BUCKET, 'data_acquisition/seesaw/preprocessed/sr_venues/cummulative_seesaw_venues')
    if 'Unnamed: 0' in dfss.columns:
        del dfss['Unnamed: 0']
    dfss = dfss.rename(columns={'venue_id': 'seesaw_venue_id'})
    dfss['seesaw_venue_id'] = pd.to_numeric(dfss['seesaw_venue_id'], errors='coerce')
    dfss = dfss.dropna(subset=['venue_name'])
    dfss['museum'] = (
        dfss.venue_name.str.contains('museum', regex=False, case=False) &
        ~dfss.venue_name.str.contains('gallery', regex=False, case=False)
    )
    dfss['museum'] += dfss.venue_name.str.contains('🏛️')
    dfss['venue_type'] = np.where(dfss.museum > 0, 'Museum', 'Gallery')
    del dfss['museum']
    dfss.venue_name = dfss.venue_name.apply(lambda x: emoji.replace_emoji(x, ''))
    dfss['clean_name'] = dfss.venue_name.str.lower().str.replace('gallery', '', regex=False).str.strip()
    logging.info("Loaded seesaw venues: %s", len(dfss))

    # --- Load and prepare artforum venues ---
    dfag = get_newest_file(s3, BUCKET, 'data_acquisition/artforum/preprocessed/sr_venues/artguide_venues')
    dfag = dfag.rename(columns={'venue_id': 'ag_venue_id'})
    dfag['ag_venue_id'] = pd.to_numeric(dfag['ag_venue_id'], errors='coerce')
    dfag.venue_name = dfag.venue_name.str.split('|').str[0].str.strip()
    dfag['clean_name'] = dfag.venue_name.str.lower().str.replace('gallery', '', regex=False).str.strip()
    dfag['venue_type'] = 'Gallery'
    logging.info("Loaded artforum venues: %s", len(dfag))

    # --- Load and prepare artsy venues ---
    dfay = get_newest_file(s3, BUCKET, 'data_acquisition/artsy/preprocessed/sr_venues/artsy_venues')
    dfay = dfay.rename(columns={'venue_id': 'artsy_venue_id'})
    dfay['artsy_venue_id'] = pd.to_numeric(dfay['artsy_venue_id'], errors='coerce')
    dfay.venue_name = dfay.venue_name.str.split('|').str[0].str.strip()
    dfay['clean_name'] = dfay.venue_name.str.lower().str.replace('gallery', '', regex=False).str.strip()
    dfay['venue_type'] = 'Gallery'
    dfay.metro_area = dfay.metro_area.str.title()
    dfay = dfay.dropna(subset=['lat', 'long'])
    logging.info("Loaded artsy venues: %s", len(dfay))

    # --- Load and prepare artrabbit venues ---
    dfrab = get_newest_file(s3, BUCKET, 'data_acquisition/artrabbit/preprocessed/sr_venues/artrabbit_venues')
    dfrab = dfrab.rename(columns={'venue_id': 'rabbit_venue_id'})
    dfrab['rabbit_venue_id'] = pd.to_numeric(dfrab['rabbit_venue_id'], errors='coerce')
    dfrab = dfrab.dropna(subset=['rabbit_venue_id'])
    dfrab['clean_name'] = dfrab.venue_name.str.lower().str.replace('gallery', '', regex=False).str.strip()
    logging.info("Loaded artrabbit venues: %s", len(dfrab))

    # --- Sequential source merges ---
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['long'] = pd.to_numeric(df['long'], errors='coerce')
    dfss['lat'] = pd.to_numeric(dfss['lat'], errors='coerce')
    dfss['long'] = pd.to_numeric(dfss['long'], errors='coerce')
    dfag['lat'] = pd.to_numeric(dfag['lat'], errors='coerce')
    dfag['long'] = pd.to_numeric(dfag['long'], errors='coerce')
    dfay['lat'] = pd.to_numeric(dfay['lat'], errors='coerce')
    dfay['long'] = pd.to_numeric(dfay['long'], errors='coerce')
    dfrab['lat'] = pd.to_numeric(dfrab['lat'], errors='coerce')
    dfrab['long'] = pd.to_numeric(dfrab['long'], errors='coerce')

    logging.info("Merging seesaw venues, prior count: %s", len(df))
    df = make_update(df, dfss, 'seesaw_venue_id')
    logging.info("After seesaw merge: %s venues", len(df))

    logging.info("Merging artforum venues")
    df = make_update(df, dfag, 'ag_venue_id', 1)
    logging.info("After artforum merge: %s venues", len(df))

    logging.info("Merging artsy venues")
    df = make_update(df, dfay, 'artsy_venue_id', 1)
    logging.info("After artsy merge: %s venues", len(df))

    logging.info("Merging artrabbit venues")
    df = make_update(df, dfrab, 'rabbit_venue_id', 0)
    logging.info("After artrabbit merge: %s venues", len(df))

    # --- Load and inject manual Google Sheet venues ---
    GSHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSlwhE5Opsbf7S7BAg1Ldf4Xy6muaSQ_5vIqnw-TWDYuuoRAaH3vsH-YXOUVUP5zBgLGVlmqLYtdg2T/pub?gid=0&single=true&output=csv"
    dfvf = pd.read_csv(GSHEET_URL)
    dfvf['venue_id'] = dfvf['venue_id'] + 2000000 if 'venue_id' in dfvf.columns else [i + 2000001 for i in range(len(dfvf))]
    dfvf['state'] = 'NY'
    dfvf['country'] = 'US'
    dfvf['metro_area'] = 'New York'
    dfvf['timezone'] = 'America/New_York'
    dfvf['clean_name'] = dfvf.venue_name.str.lower().str.split(',').str[0].str.replace('gallery', '', regex=False).str.strip()
    df = pd.concat([df, dfvf])
    logging.info("After manual sheet injection: %s venues", len(df))

    # --- Org ID assignment (incremental — preserves existing org_ids) ---

    # Merge existing venue→org mappings back onto df
    df = df.merge(dfvo, on='venue_id', how='left')

    # Split: venues with an existing org_id vs those that need one
    dfold = df[~df['org_id'].isnull()]
    df = df[df['org_id'].isnull()]

    # Derive org_name from venue_name
    df['org_name'] = df['venue_name'].apply(html.unescape)
    df['org_name'] = df['org_name'].str.lower().str.replace('\u200b', '')
    df['org_name'] = df['org_name'].str.split(', ').str[0]
    df['org_name'] = df['org_name'].str.split('(').str[0]
    df['org_name'] = df['org_name'].str.split('|').str[0]
    df['org_name'] = df['org_name'].str.replace('[ ]+',' ', regex=True)
    df['org_name'] = df['org_name'].str.strip()
    df['org_name'] = df['org_name'].str.replace('gallery', '')
    df['org_name'] = df['org_name'].str.replace('galerie', '')
    df['org_name'] = df['org_name'].str.replace('galeria', '')
    df['org_name'] = df['org_name'].str.replace('galleria', '')
    df['org_name'] = df['org_name'].str.replace('annex', '')
    df['org_name'] = df['org_name'].str.replace('project room', '')
    df['org_name'] = df['org_name'].str.replace('new york|montauk|east broadway|art center', '', regex=True)
    df['org_name'] = df['org_name'].str.replace('[ ]+',' ', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace('window$', '', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace('fine arts$', '', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace('fine art$', '', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace(' arts$', '', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace(' art$', '', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace('shop$', '', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace('projects$', '', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace('prints$', '', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace('contemporary$', '', regex=True)
    df['org_name'] = df['org_name'].str.strip().str.replace('^the ', '', regex=True)
    df['org_name'] = df['org_name'].str.strip()
    df['org_name'] = np.where(df['org_name'] == '', df['clean_name'], df['org_name'])

    # Lowercase dfo org_names and add cross-join key
    dfo['org_name'] = dfo['org_name'].str.lower()
    dfo['key'] = 1

    # Exact match against org table
    del df['org_id']
    df = df.merge(dfo[['org_name', 'org_id']], on='org_name', how='left')
    dfnew = pd.concat([dfold, df[~df['org_id'].isnull()]])
    df = df[df['org_id'].isnull()]
    del df['org_id']
    logging.info("Exact org matches: %s, still unmatched: %s", len(dfnew) - len(dfold), len(df))

    # Prefix match: cross-join remaining venues against org table
    df['key'] = 1
    dfm = df.merge(dfo, on='key', suffixes=('', '2'))
    if len(dfm) > 0:
        dfm['org_name'] = ' ' + dfm['org_name'] + ' '
        dfm['org_name2'] = ' ' + dfm['org_name2'] + ' '
        dfm['s2'] = dfm.apply(lambda row: row['org_name'].startswith(row['org_name2']), axis=1)
        dfm['l'] = dfm['org_name2'].str.len()
        dfm = dfm[
            dfm['s2'] &
            (dfm['l'] >= 9) &
            (~dfm['org_name2'].str.strip().isin(['chelsea', 'american', 'atlantic', 'broadway', 'national', 'high line', 'center for']))
        ]
    if len(dfm) > 0:
        dfm['org_name'] = dfm['org_name'].str.strip()
        dfm['org_name2'] = dfm['org_name2'].str.strip()
        # Pick the longest (most specific) org table entry that matches
        dfm = dfm.sort_values(['venue_id', 'l']).groupby('venue_id').head(1)
        dfm['org_name'] = dfm['org_name2']
        dfm = dfm[dfnew.columns]
        dfnew = pd.concat([dfnew, dfm])
        df = df[~df['venue_id'].isin(dfm['venue_id'])]
    logging.info("Prefix org matches: %s, still unmatched: %s", len(dfm), len(df))

    # Assign new org_ids to anything still unmatched
    dfo_new = df[['org_name']].drop_duplicates()
    dfo_new = dfo_new.reset_index()
    dfo_new.columns = ['org_id', 'org_name']
    dfo_new['org_id'] += dfo['org_id'].max() + 1
    df = df.merge(dfo_new, on='org_name')
    del df['key']
    logging.info("New orgs assigned: %s", len(dfo_new))

    # Combine all groups and clean up
    dfnew = pd.concat([dfnew, df])
    del dfnew['org_name']

    # Update org table with new orgs
    dfo = pd.concat([dfo, dfo_new])
    del dfo['key']
    dfo['org_name'] = dfo['org_name'].str.title()
    if 'bio' not in dfo.columns:
        dfo['bio'] = ''

    df = dfnew

    # --- Write outputs ---
    write_file(s3, df, BUCKET, 'showrunner/merge/', 'venues')
    write_file(s3, dfo, BUCKET, 'showrunner/merge/', 'org')
    logging.info("=== PP5 venues merge complete ===")


if __name__ == "__main__":
    main()
