# DS 4320 Project 2 - Predicting Election Results

### Executive Summary

This project investigates whether consumer-facing economic indicators (specifically gasoline prices and inflation) are statistically associated with incumbent party performance in U.S. presidential elections from 1976 to 2020. Motivated by the theory of economic voting, which holds that voters reward or punish the party in power based on how the economy feels in their daily lives, the analysis combines three publicly available datasets: state-level presidential election returns from the MIT Election Data Science Lab, monthly retail gasoline prices from the Federal Reserve Economic Data (FRED), and monthly Consumer Price Index data from FRED. These sources were merged at the state-year level, producing approximately 600 documents stored in a MongoDB Atlas collection, and analyzed using correlation analysis and both simple and multiple linear regression. The findings suggest that both gas price changes and inflation rate are negatively associated with incumbent vote share. So, when prices rise, the party in power tends to lose ground at the polls, though the strength of this relationship varies across election cycles and is moderated by factors including political polarization, third-party candidacies, and regional economic variation.

---
### Name - Rachel Seo
### NetID - ydp7xv
### DOI - [https://doi.org/10.5281/zenodo.19356307](https://doi.org/10.5281/zenodo.19356307)
### Press Release
[**At the Pump and At the Polls**](https://github.com/rachelsseo/pumps-to-polls/blob/main/press-release.md)
### Data - [Link to Data in OneDrive](https://myuva-my.sharepoint.com/:f:/g/personal/ydp7xv_virginia_edu/IgABdGPnCfnZTreqtZTAhl15AbNbqImbOeT5r1bYr74uxzo?e=aBlyP6)
### Pipeline - [Link to Analysis Code](https://github.com/rachelsseo/pumps-to-polls/blob/main/scripts/full-pipeline.ipynb)
### License - [MIT](https://github.com/rachelsseo/pumps-to-polls/blob/main/LICENSE.md)
---

| Spec | Value |
|---|---|
| Name | Rachel Seo |
| NetID | ydp7xv |
| DOI | [https://doi.org/10.5281/zenodo.19356307](https://doi.org/10.5281/zenodo.19356307) |
| Press Release | [At the Pump and at the Polls](https://github.com/rachelsseo/pump-to-polls/blob/main/press-release.md) |
| Data | [Link to Data in OneDrive](https://myuva-my.sharepoint.com/:f:/g/personal/ydp7xv_virginia_edu/IgABdGPnCfnZTreqtZTAhl15AbNbqImbOeT5r1bYr74uxzo?e=aBlyP6) |
| Pipeline | [Link to Analysis Code](https://github.com/rachelsseo/pumps-to-polls/blob/main/scripts/full-pipeline.ipynb)|
| License | [MIT](https://github.com/rachelsseo/pump-to-polls/blob/main/LICENSE.md) |

---

## Problem Definition
### General and Specific Problem
* **General Problem:** Voters are often influenced by how they perceive/feel economic conditions seem to be impacting their everyday. So, how do economic conditions in the months leading up to a presidential election affect the incumbent party's share of the two-party popular vote?
* **Specific Problem:** How do fluctuating gas prices and inflation rates in the months leading up to a presidential election affect the incumbent party's share of the two-party popular vote?

### Rationale
Voters often feel the everyday economic impacts through changing gas prices and inflation rates. As a college student/everyday citizen, I know that I feel the economic effects personally through fluctuating gas prices the most. So, this project investigates whether changes in gas prices and inflation in the months leading up to a presidential election are statistically associated with the incumbent party's share of the two-party popular vote from 1976 to 2020.

### Motivation
Economic conditions are widely believed to influence electoral outcomes, yet the precise relationship between consumer-facing indicators (gasoline prices and inflation) and incumbent party performance remains difficult to quantify. Through this project, I want to explore the relationship between these indicators and the electoral outcomes - just based on curiousity. Although my data only spans from 1976 to 2020, it will be interesting to see what patterns and trends emerge and whether they will continue on in the next presidential election.

### Press Release Headline and Link
[**At the Pump and at the Polls**](https://github.com/rachelsseo/pumps-to-polls/blob/main/press-release.md)

## Domain Exposition

*Summary Table*

| Term | Definition |
|---|---|
| U.S. Bureau of Labor Statistics | The principal fact-finding agency for the Federal Government in the broad field of labor economics and statistics - where FRED pulls data from. |
| FRED | Federal Reserve Economic Data; data on monthly CPI (inflation) and average annual gas prices going back to the 1970s |
| Incumbent| The current holder of political office |
| CPI | Consumer Price Index, numerically representation of inflation |
| Popular vote | Decision made by voters reflected in the share of votes won by a particular candidate, party or option in a referendum.|

*Subject Domain*

This project lives in the intersection of political science, economics, and data science. More specifically, this project dives deeper into the field of electoral studies within political science, it examines voting behavior, party performance, and what drive election outcomes. We draw upon macroeconomic concepts as we use CPI and gas prices as explanatory variables. The application portion/methodology - regression modeling & correlation analysis - of the research is firmly data science. 


*Background Reading*

[Link to background readings](https://myuva-my.sharepoint.com/:f:/g/personal/ydp7xv_virginia_edu/IgABdGPnCfnZTreqtZTAhl15AbNbqImbOeT5r1bYr74uxzo?e=oOGMY4)

| Title | Brief Description | Link to File |
|-------|------------------|------|
| Economic Perceptions and Voting Behavior in US Presidential Elections" |This study demonstrates that voters' objective perceptions of the economy (independent of partisan bias) have a measurable and significant effect on whether they support or oppose the incumbent presidential candidate.| [Link](https://myuva-my.sharepoint.com/:b:/g/personal/ydp7xv_virginia_edu/IQA9H2L3UtyVSYIsSv7nK54uAXnbGq8ZZWVYUBc9ilalJpU) |
|"Presidential Approval and Gas Prices"|This study finds that rising gas prices directly reduce presidential approval ratings, that this effect operates through voters' personal economic experiences rather than media coverage.| [Link](https://myuva-my.sharepoint.com/:b:/r/personal/ydp7xv_virginia_edu/Documents/Project%202/background%20readings/US%20ELECTIONS_%20Gasoline%20price%20fluctuations%20could%20play%20role%20in%202024%20presidential%20election%20_%20S%26P%20Global.pdf?csf=1&web=1&e=NdNk0x)|
|"Gasoline Prices and Presidential Approval Ratings of the United States"| Using random forests, this study finds that the relationship between gas prices and presidential approval is nonlinear and retains out-of-sample predictive power even after controlling for broader macroeconomic indicators.| [Link](https://myuva-my.sharepoint.com/:b:/r/personal/ydp7xv_virginia_edu/Documents/Project%202/background%20readings/gupta-et-al-2025-gasoline-prices-and-presidential-approval-ratings-of-the-united-states.pdf?csf=1&web=1&e=MFDJkz)) |
|"Is It Still the Economy? Economic Voting in Polarized Politics"|This paper argues that rising political polarization has weakened economic voting, as partisan loyalty increasingly causes voters to stick with their party even when economic conditions would historically have driven them to defect. | [Link](https://myuva-my.sharepoint.com/:b:/r/personal/ydp7xv_virginia_edu/Documents/Project%202/background%20readings/is-it-still-the-economy-economic-voting-in-polarized-politics.pdf?csf=1&web=1&e=Qg115b)|
|"The Impact of Job Growth and Inflation on Presidential Voting Patterns"|Across 17 presidential elections from 1956 to 2020, this study finds that CPI changes measured nine months before Election Day, rather than GDP or unemployment, are the strongest predictors of incumbent party vote share.| [Link](https://myuva-my.sharepoint.com/:b:/r/personal/ydp7xv_virginia_edu/Documents/Project%202/background%20readings/ssrn-4657592.pdf?csf=1&web=1&e=Rv7SpJ)|



## Data Creation 

### Provenance 
The dataset used in this analysis was constructed by merging three publicly available data sources. Presidential election returns were sources from the MIT Election Data Science Lab's U.S. President 1976-2020 dataset. This dataset aggregates state-level candidate vote totlas across all major and minor parties for each presidential cycle.
Annual average retail gas prices were downloaded from the Federal Reserve Economic Data (FRED) and so was the inflation data. The inflation data was drawn via the Consumer Price Index for All Urban Consumers.

All three datasets were first downloaded in CSV format, cleaned and aggregated to the election-yera level in Python using pandas, merged on the shared `year` key, and loaded into a MongoDB Atlas collection.


### Code 
| File | Description | Link |
|---|---|---|
| data-creation.py | Combined and cleaned dataset merging the three sources above at the state-year level, with incumbent vote share, gas price % change, and inflation rate per election year, loaded into MongoDB Atlas | [Link in Code](https://github.com/rachelsseo/pumps-to-polls/blob/main/scripts/data-creation.py) |

### Bias Identification 

Several sources of bias may have been introduced during both the raw data collection and merging process. Since the MIT Election Data Science Lab dataset aggregates official state-certified vote totals, this means any systemic issues in vote reporting are inherited directly into analysis.

The decision to calculate incumbent vote share against *all* votes rather than the strict two-party vote introduces deflation bias in years with strong third-party candidates.

Using national average gas prices and a single CPI figure for all states asummes that every state experienced the same economic conditions in a given election year. This ignores meaningful regional variation - a voter in rural Montana paying high prices to commute long distances is treated identically to an urban voter who has access to public transit.

### Bias Mitigation 

In order to mitigate the sources of bias identified in the data collection process, many steps were taken. To address inconsistencies in third-party vote reporting, incumbent vote share was calculated consistently across all election years using the same formula (total incumbent party votes/total votes cast). This ensured that no election year was treated differently from another, even if the resulting figures are not directly coparable to published tow-party vote share statistics.

State-level details were preserved in the final dataset rather than collapsing immediately to national averages, which partially addresses the regional economic variation problem by allowing downstream analysis to examine how the relationship between gas prices, inflation, and incumbent performance differes across states.

Both the gas price and CPI variables are clearly documented as national averages sources from FRED, so any analysis using these figures should be interpreted with the understanding that they represent broad economic conditions rather than localized voter experiences.

### Rationale 

The decision to map incumbent party by election year rather than by candidate was a deliberate decision since it aligns with the theoretical framework of economic voting theory (holds that voters reward or punish the party in power rather than any individual candidate).

Aggregating monthly gas prices and CPI values into a single annual average for each election year was chosen for simplicity - so this may understate the true relationship between economic conditions and electoral outcomes.
Using states as the unit of analysis rather than the county or individual level was a practical jugement because of data availability, but this also asummes that the average economic conditions and average vote share within a state reflect the same underlying relationship that would be observed if individual-level data were available.


## Metadata

### Implicit Schema
- Unit of observation: each document in the MongoDB collection represents a single U.S. state in a single presidential year - so the natural key is a combination of `year` and `state_po`

- Economic indicators are year-level, not state level: `gas_price_change_pct` and `inflation_rate` are national averages that repeat identically across all 51 states rows within a given election year 

- Incumbent party is a derived categorical variable: `incumbent_party` field is not sourced directly from any of the three raw datasets but is manually encoded using domain knowledge about which party held the White House in each election year

- Vote share is a computed ratio, not a raw count: `incumbent_vote_share` is a percentage derived from the raw `candidatevotes` totals in the MIT dataset - so it reflects the incumbent party's share of *all* votes cast rather than the two-party vote share, and downstream visualizations or models should treat it accordingly 

### Data Tables 

| Table Name | Description | Link |
|---|---|---|
| `1976-2020-president.csv` | State-level U.S. presidential election returns for all candidates across 12 election cycles from 1976 to 2020, sourced from the MIT Election Data Science Lab | [1976-2020-president.csv](https://github.com/rachelsseo/pumps-to-polls/blob/main/data/1976-2020-president.csv) |
| `avg_gas_price.csv` | Monthly U.S. city average retail price of unleaded regular gasoline per gallon (FRED series APU000074714) from January 1976 to January 2020 | [avg_gas_price.csv](https://github.com/rachelsseo/pumps-to-polls/blob/main/data/avg_gas_price.csv) |
| `cpi.csv` | Monthly Consumer Price Index for All Urban Consumers (FRED series CPIAUCSL) from January 1976 to January 2020, used to derive annual inflation rates | [cpi.csv](https://github.com/rachelsseo/pumps-to-polls/blob/main/data/cpi.csv) |
| `election_economics` | Combined MongoDB collection merging all three sources above at the state-year level, containing incumbent vote share, gas price percent change, and inflation rate per election year per state | [data-creation.py](https://github.com/rachelsseo/pumps-to-polls/blob/main/scripts/data-creation.py) |

### Data Dictionary 

| Feature | Data Type | Description | Example |
|---|---|---|---|
| `year` | `int` | The presidential election year | `1980` |
| `state` | `str` | Full name of the U.S. state or district | `ALABAMA` |
| `state_po` | `str` | Two-letter postal abbreviation of the state | `AL` |
| `incumbent_party` | `str` | The political party holding the White House in the given election year, manually encoded from domain knowledge | `DEMOCRAT` |
| `incumbent_vote_share` | `float` | The incumbent party's share of all votes cast in that state in that election year, expressed as a percentage | `47.45` |
| `gas_price_change_pct` | `float` | Year-over-year percentage change in the national average retail price of unleaded regular gasoline (FRED APU000074714), averaged across all months in the election year | `34.21` |
| `inflation_rate` | `float` | Year-over-year percentage change in the Consumer Price Index for All Urban Consumers (FRED CPIAUCSL), averaged across all months in the election year | `13.55` |

### Uncertainty Quantification 

| Feature | Table | Type | Description | Uncertainty Notes |
|---|---|---|---|---|
| `incumbent_vote_share` | `election_economics` | `float` | Incumbent party's share of all votes cast in a given state and election year, expressed as a percentage | Sensitive to third-party candidates — 1992 (Perot) artificially deflates both major party shares; calculated against all votes rather than two-party vote which reduces comparability across years |
| `gas_price_change_pct` | `election_economics` | `float` | Year-over-year percentage change in the national average retail price of unleaded regular gasoline, averaged across all months in the election year | High variance driven by oil shock years (1979–1980) and the 2008 financial crisis collapse; national average masks significant regional price variation; missing for 2020 (~51 documents) |
| `inflation_rate` | `election_economics` | `float` | Year-over-year percentage change in the Consumer Price Index for All Urban Consumers, averaged across all months in the election year | Skewed right by high-inflation years of 1979–1981; full-year average may understate electoral impact compared to a pre-election window as suggested by Doti & Campbell (2023); missing for 2020 (~51 documents) |
| `incumbent_party` | `election_economics` | `str` | Political party holding the White House in the given election year | Manually encoded judgment-dependent field — uncertainty arises in open-seat elections (2000, 2008) where voters may not attribute economic conditions to the incumbent party's candidate |
| `year` | `election_economics` | `int` | Presidential election year | No uncertainty — discrete fixed values every four years from 1976 to 2020 |
| `state` | `election_economics` | `str` | Full name of the U.S. state or district | No uncertainty — sourced directly from MIT Election Data Science Lab certified returns |
| `state_po` | `election_economics` | `str` | Two-letter postal abbreviation of the state | No uncertainty — standard USPS abbreviations derived directly from the MIT dataset |