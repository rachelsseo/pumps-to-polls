import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()  # also prints to terminal
    ]
)
log = logging.getLogger(__name__)

load_dotenv()

# 1. Election Data
try:
    elections = pd.read_csv("data/1976-2020-president.csv")
    log.info(f"Loaded election data: {len(elections)} rows")
except FileNotFoundError:
    log.error("Election CSV not found — check your file path")
    raise

incumbent_map = {
    1976: "REPUBLICAN", 1980: "DEMOCRAT",   1984: "REPUBLICAN",
    1988: "REPUBLICAN", 1992: "REPUBLICAN", 1996: "DEMOCRAT",
    2000: "DEMOCRAT",   2004: "REPUBLICAN", 2008: "REPUBLICAN",
    2012: "DEMOCRAT",   2016: "DEMOCRAT",   2020: "REPUBLICAN"
}

election_years = list(incumbent_map.keys())
results = []
skipped = 0

for year in election_years:
    df_year = elections[elections["year"] == year]
    incumbent_party = incumbent_map[year]

    for state in df_year["state"].unique():
        df_state = df_year[df_year["state"] == state]
        incumbent_votes = df_state[df_state["party_simplified"] == incumbent_party]["candidatevotes"].sum()
        other_votes = df_state[df_state["party_simplified"] != incumbent_party]["candidatevotes"].sum()
        total = incumbent_votes + other_votes

        if total == 0:
            log.warning(f"Skipping {state} {year} — total votes is 0")
            skipped += 1
            continue

        results.append({
            "year": year,
            "state": state,
            "state_po": df_state["state_po"].iloc[0],
            "incumbent_party": incumbent_party,
            "incumbent_vote_share": round(incumbent_votes / total * 100, 2)
        })

elections_df = pd.DataFrame(results)
log.info(f"Election records built: {len(elections_df)} rows, {skipped} skipped")

# 2. Gas Prices
try:
    gas = pd.read_csv("data/avg_gas_price.csv")
    log.info(f"Loaded gas price data: {len(gas)} rows")
except FileNotFoundError:
    log.error("Gas price CSV not found — check your file path")
    raise

gas.columns = ["date", "gas_price"]
gas["date"] = pd.to_datetime(gas["date"])
gas["year"] = gas["date"].dt.year

gas_annual = gas.groupby("year")["gas_price"].mean().reset_index()
gas_annual["gas_price_change_pct"] = gas_annual["gas_price"].pct_change() * 100
gas_annual = gas_annual[gas_annual["year"].isin(election_years)][["year", "gas_price_change_pct"]].round(2)
log.info(f"Gas price annual averages computed for {len(gas_annual)} election years")

# 3. CPI/Inflation
try:
    cpi = pd.read_csv("data/cpi.csv")
    log.info(f"Loaded CPI data: {len(cpi)} rows")
except FileNotFoundError:
    log.error("CPI CSV not found — check your file path")
    raise

cpi.columns = ["date", "cpi"]
cpi["date"] = pd.to_datetime(cpi["date"])
cpi["year"] = cpi["date"].dt.year

cpi_annual = cpi.groupby("year")["cpi"].mean().reset_index()
cpi_annual["inflation_rate"] = cpi_annual["cpi"].pct_change() * 100
cpi_annual = cpi_annual[cpi_annual["year"].isin(election_years)][["year", "inflation_rate"]].round(2)
log.info(f"Inflation rates computed for {len(cpi_annual)} election years")

# 4. Merge
try:
    combined = elections_df \
        .merge(gas_annual, on="year") \
        .merge(cpi_annual, on="year")
    log.info(f"Merge successful: {len(combined)} total documents")

    null_counts = combined.isnull().sum()
    if null_counts.any():
        log.warning(f"NaN values detected after merge:\n{null_counts[null_counts > 0]}")
    else:
        log.info("No NaN values found in merged dataset")

except Exception as e:
    log.error(f"Merge failed: {e}")
    raise

# 5. Upload to MongoDB
try:
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not found in .env file")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()  # force connection check
    log.info("Connected to MongoDB successfully")

    db = client["data_by_design"]
    collection = db["election_economics"]

    deleted = collection.delete_many({})
    log.info(f"Cleared {deleted.deleted_count} existing documents")

    records = combined.to_dict(orient="records")
    collection.insert_many(records)
    log.info(f"Inserted {len(records)} documents into 'election_economics'")

except ValueError as e:
    log.error(f"Configuration error: {e}")
    raise
except Exception as e:
    log.error(f"MongoDB error: {e}")
    raise

log.info("Pipeline complete")