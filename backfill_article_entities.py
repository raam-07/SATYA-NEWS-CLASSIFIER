import os
import re
import json
import logging
import sqlite3

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PARTY_SLUG_ALIASES = {
  'bharatiya_janata_party': 'bjp',
  'bhartiya_janata_party': 'bjp',
  'bharatiya_janata': 'bjp',
  'indian_national_congress': 'inc',
  'congress': 'inc',
  'congress_party': 'inc',
  'grand_old_party': 'inc',
  'aam_aadmi_party': 'aap',
  'aam_aadmi': 'aap',
  'common_man_party': 'aap',
  'all_india_trinamool_congress': 'tmc',
  'trinamool': 'tmc',
  'aitc': 'tmc',
  'trinamool_congress': 'tmc',
  'samajwadi_party': 'sp',
  'samajwadi': 'sp',
  'bahujan_samaj_party': 'bsp',
  'bahujan_samaj': 'bsp',
  'dravida_munnetra_kazhagam': 'dmk',
  'dravidam': 'dmk',
  'communist_party_of_india_marxist': 'cpm',
  'cpim': 'cpm',
  'left_front': 'cpm',
  'marxist': 'cpm',
  'janata_dal_united': 'jdu',
  'nitish_party': 'jdu',
  'nationalist_congress_party': 'ncp',
  'nationalist_congress': 'ncp',
  'telugu_desam_party': 'tdp',
  'telugu_desam': 'tdp',
  'jharkhand_mukti_morcha': 'jmm',
  'jharkhand_mukti': 'jmm',
  'rashtriya_janata_dal': 'rjd',
  'rashtriya_janata': 'rjd',
  'all_india_majlis_e_ittehadul_muslimeen': 'aimim',
  'majlis': 'aimim',
  'mim': 'aimim',
  'shiv_sena_eknath_shinde': 'shiv_sena',
  'shinde_sena': 'shiv_sena',
  'balasahebanchi_shiv_sena': 'shiv_sena',
  'viduthalai_chiruthaigal_katchi': 'vck',
  'jammu_and_kashmir_peoples_democratic_party': 'pdp',
  'peoples_democratic_party': 'pdp',
  'all_india_anna_dravida_munnetra_kazhagam': 'aiadmk',
  'all_india_anna_dmk': 'aiadmk',
  'marumalarchi_dravida_munnetra_kazhagam': 'mdmk',
}

def slugify(name):
    if not name:
        return ""
    s = name.lower()
    s = s.replace(' ', '_')
    s = s.replace('.', '')
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s

def party_slugify(name):
    s = slugify(name)
    return PARTY_SLUG_ALIASES.get(s, s)

def get_db_connection():
    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    
    if db_url and (db_url.startswith('libsql://') or db_url.startswith('https://')):
        try:
            import libsql
            return libsql.connect(database=db_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local sqlite3.")
            
    # Default fallback path
    default_db_path = '/Users/mac/Downloads/Code/Satya/satya.db'
    if not os.path.exists(os.path.dirname(default_db_path)):
        default_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'satya.db')
    db_path = os.environ.get('SATYA_DB_PATH', default_db_path)
    return sqlite3.connect(db_path)

def bulk_insert_entities(cursor, insert_data):
    if not insert_data:
        return
    chunk_size = 200
    for i in range(0, len(insert_data), chunk_size):
        chunk = insert_data[i:i+chunk_size]
        placeholders = ", ".join(["(?, ?, ?)" for _ in range(len(chunk))])
        sql = f"INSERT OR IGNORE INTO article_entities (article_id, kind, slug) VALUES {placeholders}"
        params = []
        for r in chunk:
            params.extend(r)
        cursor.execute(sql, params)

def main():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create table and index if not exists (in case it wasn't run)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS article_entities (
          article_id INTEGER NOT NULL REFERENCES articles(id),
          kind TEXT NOT NULL CHECK(kind IN ('party','minister','state','city','topic')),
          slug TEXT NOT NULL,
          PRIMARY KEY (article_id, kind, slug)
        );
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ae_kind_slug ON article_entities(kind, slug, article_id);
    """)
    conn.commit()

    # 2. Get start_id
    start_id = 0
    cursor.execute("SELECT IFNULL(MAX(article_id), 0) FROM article_entities")
    start_id = cursor.fetchone()[0]
    logging.info(f"Resuming backfill from article ID > {start_id}")
    
    total_processed = 0
    batch_size = 10000
    
    while True:
        logging.info(f"Fetching batch of {batch_size} articles starting after ID {start_id}...")
        cursor.execute("""
            SELECT id, party_mentioned, ministers_mentioned, states_mentioned, cities_mentioned, topic_tags
            FROM articles
            WHERE status IN ('classified', 'entity_processed', 'processed') AND id > ?
            ORDER BY id ASC
            LIMIT ?
        """, (start_id, batch_size))
        
        rows = cursor.fetchall()
        if not rows:
            logging.info("No more articles to process.")
            break
            
        article_ids = [r[0] for r in rows]
        insert_data = []
        
        for r in rows:
            article_id = r[0]
            party_json = r[1]
            ministers_json = r[2]
            states_json = r[3]
            cities_json = r[4]
            topics_json = r[5]
            
            def parse_list(j):
                if not j:
                    return []
                try:
                    return json.loads(j)
                except Exception:
                    if isinstance(j, list):
                        return j
                    return []
            
            parties = parse_list(party_json)
            ministers = parse_list(ministers_json)
            states = parse_list(states_json)
            cities = parse_list(cities_json)
            topics = parse_list(topics_json)
            
            kinds = [
                ('party', parties),
                ('minister', ministers),
                ('state', states),
                ('city', cities),
                ('topic', topics)
            ]
            
            for kind, items in kinds:
                if isinstance(items, list):
                    for item in set(items):
                        if not item:
                            continue
                        slug = party_slugify(item) if kind == 'party' else slugify(item)
                        if slug:
                            insert_data.append((article_id, kind, slug))
            
            start_id = article_id
            total_processed += 1
            
        # Bulk delete in one single query
        if article_ids:
            chunk_size = 200
            for i in range(0, len(article_ids), chunk_size):
                chunk = article_ids[i:i+chunk_size]
                placeholders = ", ".join(["?" for _ in range(len(chunk))])
                cursor.execute(f"DELETE FROM article_entities WHERE article_id IN ({placeholders})", chunk)
                
        # Bulk insert
        bulk_insert_entities(cursor, insert_data)
            
        conn.commit()
        logging.info(f"Committed batch of {len(rows)} articles. Total processed this run: {total_processed}")
        
    conn.close()
    logging.info("Backfill completed successfully.")

if __name__ == '__main__':
    main()
