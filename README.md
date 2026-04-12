# DS 4320 Project 1 - Allocating Emergency Response Resources 

### Executive Summary

Urban emergency medical services in metropolitan Sydney face mounting pressure from rising incident volumes, intensifying traffic congestion, and unpredictable environmental conditions that strain traditional dispatch systems. This project leverages the Integrated Emergency Response Analytics Dataset (IERAD) - a synthetic, multi-source dataset spanning 2018 to 2024 - to develop a data-driven, AI-augmented decision-support framework for optimizing emergency resource allocation across Sydney's urban and suburban network. The dataset was restructured into a normalized relational schema of four tables capturing incident characteristics, environmental conditions, resource availability, and dispatch coordination, enabling more rigorous and reproducible analysis. Two predictive models were built: a Gradient Boosting classifier that recommends the optimal dispatch type (Ambulance Only, Drone Only, or Hybrid), and a regression model that estimates response time to support real-time routing decisions. The system is explicitly designed as a human-in-the-loop tool, surfacing model confidence alongside recommendations so that dispatch coordinators retain final decision-making authority. Key limitations include the synthetic nature of the underlying data, potential class imbalance in dispatch outcomes, and the absence of real-time road network routing, which are all identified as priorities for future development. Situated at the intersection of emergency management, transportation logistics, urban air mobility, and AI decision-support systems, this project demonstrates the feasibility of machine learning as a meaningful complement to human judgment in high-stakes, time-critical public safety operations.

---

### Name - Rachel Seo
### NetID - ydp7xv
### DOI - [https://doi.org/10.5281/zenodo.19356307](https://doi.org/10.5281/zenodo.19356307)
### Press Release
[**🚨 Have An Emergency? Help Might Already Be on the Way**](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/press-release.md)
### Data - [Link to Data in OneDrive](https://myuva-my.sharepoint.com/:f:/g/personal/ydp7xv_virginia_edu/IgCWyGzL4V5tQqkgTka4so5TAezqX4OrvqBxwWAb1dqnsx4?e=6AMQEc)
### Pipeline - [Link to Analysis Code](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/scripts/final-solution-pipeline.ipynb)
### License - [MIT](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/LICENSE.md)
---
| Spec | Value |
|---|---|
| Name | Rachel Seo |
| NetID | ydp7xv |
| DOI | [https://doi.org/10.5281/zenodo.19356307](https://doi.org/10.5281/zenodo.19356307) |
| Press Release | [🚨 Have An Emergency? Help Might Already Be on the Way](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/press-release.md) |
| Data | [Link to Data in OneDrive](https://myuva-my.sharepoint.com/:f:/g/personal/ydp7xv_virginia_edu/IgCWyGzL4V5tQqkgTka4so5TAezqX4OrvqBxwWAb1dqnsx4?e=6AMQEc) |
| Pipeline | [Link to Analysis Code](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/scripts/final-solution-pipeline.ipynb)|
| License | [MIT](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/LICENSE.md) |

---

## Problem Definition
### General and Specific Problem
* **General Problem:** Emergency response systems must decide how to deploy limited resources (e.g., ambulances and medical drones) under varying conditions such as traffic, weather, and incident severity, yet these decisions are often made under time pressure and uncertainty.
* **Specific Problem:** Given environmental, operational, and incident-related factors (e.g., traffic congestion, weather conditions, resource availability, and incident severity), which dispatch strategy (drone only, ambulance only, or hybrid) should be selected to optimize emergency response effectiveness?

### Rationale
After observing the Integrated Emergency Response Dataset (IERAD) (from [Kaggle](https://www.kaggle.com/datasets/datasetengineer/integrated-emergency-response-dataset-ierad)), the refinement of the problem statement is necessary because it aligns the problem with the variables available in the IERAD dataset, such as traffic congestion, weather conditions, incident severity, and resource availability. By narrowing the focus to predicting the most effective dispatch strategy, the problem becomes machine learning classification task that can support more efficient and data-driven emergency response decision-making.

### Motivation
The motivation for this project comes from wanting to support more efficient data-driven emergency response decision-making. Making emergency decision under time constraints is stressful when it involves many changing factors such as traffic congestion, weather conditions, and the availability of medical resources. As new technologies like medical drones become integrated into emergency systems, determining when to deploy drones, ambulances, or a combination of both becomes increasingly complex.

### Press Release Headline and Link
[**🚨 Have An Emergency? Help Might Already Be on the Way**](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/press-release.md)

## Domain Exposition

*Summary Table*
| Term | Full Name | Definition |
|---|---|---|
| IERAD | Integrated Emergency Response Analytics Dataset | Curated synthetic dataset of emergency response records from metropolitan Sydney (2018–2024), used to develop AI-augmented dispatch and routing models. |
| Dispatch | Emergency Dispatch | The process of assigning and directing emergency resources (ambulances, drones) to an incident location. |
| Label | Dispatch Outcome Label | Classification target variable indicating the type of resource dispatched: Ambulance Only, Drone Only, or Hybrid Dispatch. |
| Hybrid Dispatch | Hybrid Emergency Dispatch | A dispatch strategy that deploys both an ambulance and a drone simultaneously to an incident. |
| Response Time | Emergency Response Time | Time in minutes from dispatch to arrival at the incident location. Primary regression target in the predictive model. |
| Incident Severity | Incident Severity Level | Categorical classification of how serious an incident is: Low, Medium, or High. |

This project sits at the intersection of emergency management, transportation logistics, and AI decision-support systems. Within emergency management, it draws on the operational principles of emergency medical services (EMS), including triage prioritization, resource dispatching, and response time optimization — domains traditionally governed by public health and first-responder policy frameworks. On the logistics side, it engages with real-time vehicle routing and multi-modal transport optimization, particularly the emerging challenge of integrating unmanned aerial vehicles (drones) alongside ground-based ambulances in time-critical scenarios. The AI component places this work within the broader field of machine learning for public safety, specifically the subfield of human-AI collaborative decision-making, where the goal is not to replace human judgment but to augment it under uncertainty and time pressure. More niche domains it touches include urban air mobility (UAM), which governs the regulatory and operational frameworks for drone deployment in populated airspace, and algorithmic fairness in public services, since resource allocation models trained on historical data risk encoding and perpetuating disparities in response quality across different geographic regions or community types.

*Background Reading*

[Link to Background Readings](https://myuva-my.sharepoint.com/:f:/g/personal/ydp7xv_virginia_edu/IgBhxYftL9jnQpWUgzgKWH3SATp3oSA08zW2oDO7o8InUUQ?e=fR9mRh)

| Title | Brief Description | Link to File |
|-------|------------------|------|
| IERAD Data Card from Kaggle |Public description, schema overview, and basic usage notes for the same dataset you describe (or a very close precursor), including dataset integration of ambulance dispatch, drone operators, and hospitals| [Link](https://myuva-my.sharepoint.com/:w:/g/personal/ydp7xv_virginia_edu/IQAd8yhJrB7HSYucZwEjSsMKAZl_RG4dA-ZYONg_-i5b6Xo?e=pDLHxC) |
|Drones for emergency services: a whole-of-government approach to crisis prevention, response and recovery| Discusses policy, coordination, and technical benefits of drone integration for emergency services across Australia, including both proactive monitoring and reactive response use cases | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/ydp7xv_virginia_edu/IQAwuP9wSkYHQZpKXwSp1bUnAcV2k9wJFE6U4MSvn_LHKJw?e=rCeymr) |
|Ambulance route optimization in a mobile ambulance dispatch system using deep neural network (DNN)| Evaluates a mobile ambulance dispatch and route optimization model that explicitly incorporates traffic conditions, ambulance availability, patient priority, and demand prediction | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/ydp7xv_virginia_edu/IQCtL5BfrsFeQKGYBvKGU0LCAYJXUJKwyTuec27MQMv-8Bk?e=tXrSyb) |
|Drones trialled to enhance NSW Ambulance Aeromedical and Special Operations| Describes NSW Ambulance's Remotely Piloted Aeromedical Clinical Systems (RPACS) trial, including operational goals (search and rescue, supply delivery), real-time streaming, and integration with existing workflows in NSW. | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/ydp7xv_virginia_edu/IQAHYQV01VyBSLRyLBGqQfm8AUeD9kDLWSfp6gTrj2AaRlc?e=appZ8N) |
|AI-Driven Optimization of Emergency Medical Services|Presents an AI-enhanced ambulance management system using real-time geographical data for routing and coordination between field units and hospitals.| [Link](https://myuva-my.sharepoint.com/:b:/g/personal/ydp7xv_virginia_edu/IQBClPd6TBjkRL0PdpDF2h2YAVDjNG1nj0xnkkz6OOJ2Cps?e=A5jUPm) |

## Data Creation 

### Provenance 
The raw data acquisition process for this project began with a Google search for "datasets emergency response resources," followed by clicking through multiple recommended websites to evaluate available options. After reviewing several sources, the IERAD dataset was identified on Kaggle as the most suitable foundation for this project. The dataset was then modified — restructured into a normalized relational schema and augmented with surrogate keys — to create the version used in analysis and modeling.

### Code 
| File | Description | Link |
|---|---|---|
| `data-creation.py` | Loads the raw CSV, assigns a surrogate primary key, and splits the flat file into 4 normalized parquet tables: `incident`, `environmental_conditions`, `resource`, and `dispatch`. | [data-creation.py](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/data-creation.py)|

### Bias Identification 
The data acquisition process began with a Google search, which introduces algorithmic ranking bias since search engines favor popular, well-indexed, English-language sources, making datasets from underrepresented regions or smaller institutions less discoverable. Recommendation systems reflect the engagement patterns of prior users rather than dataset quality or representativeness. Sourcing the dataset from Kaggle introduces platform bias, since Kaggle's upvoting system amplifies datasets that are clean and broadly appealing rather than operationally realistic. Most critically, the IERAD dataset is synthetic, meaning its distributions of incident types, response times, and dispatch decisions reflect the assumptions of whoever generated it rather than real-world emergency operations in Sydney. Finally, modifying the dataset to create a "new" version introduces researcher bias, as decisions about table splits, column selection, and transformations are subjective judgment calls.

### Bias Mitigation 
Since the IERAD dataset is synthetic, the most important mitigation step is transparency. Clearly documenting that all findings are exploratory and should not be generalized to real Sydney emergency operations without validation against actual dispatch records is imperative. Class imbalance across the Label target variable (Ambulance Only, Drone Only, Hybrid) should be quantified and addressed through stratified sampling or class weighting during model training to prevent the classifier from simply predicting the majority class. Algorithmic bias in the dispatch classifier can be audited by disaggregating model performance metrics across subgroups such as Region_Type and Incident_Severity, checking whether the model performs equally well for rural versus urban incidents. Any researcher-introduced bias from the data modification step should be mitigated by version-controlling all transformation code so that every decision is traceable and reproducible.

### Rationale 
Splitting the flat CSV into four normalized tables was a deliberate design choice to support relational querying, though incorrect joins introduce risk of data loss or duplication. Gradient Boosting was chosen over simpler models due to the mix of categorical and numeric features, but this sacrifices interpretability — which is a meaningful tradeoff in a life-safety context. Using Label as the classification target assumes every recorded dispatch outcome was correct, which is uncertain given that many decisions were made by human coordinators with context not captured in the data.

## Metadata
### ER Diagram
![ER Diagram](/images/ERD.png)

### Data Tables 
| Table Name | Description | Link |
|---|---|---|
| `incident` | Core event records including incident type, severity, emergency level, region, injuries, distance, response time, and dispatch outcome label. One row per incident. | [incident.parquet](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/parquet-data/incident.parquet) |
| `environmental_conditions` | Weather and traffic conditions recorded at the time of each incident, including weather impact severity and air traffic level. | [environmental_conditions.parquet](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/parquet-data/environmental_conditions.parquet) |
| `resource` | Drone and ambulance availability and performance metrics per incident, including speeds, battery life, payload weight, and fuel level. | [resource.parquet](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/parquet-data/resource.parquet) |
| `dispatch` | Dispatch coordination details per incident, including coordinator type (AI or Human), specialist availability, and destination hospital capacity. | [dispatch.parquet](https://github.com/rachelsseo/emergency-resource-allocation/blob/main/parquet-data/dispatch.parquet) |

### Data Dictionary 
`incident`

| Feature Name | Data Type | Description | Example |
|---|---|---|---|
| incident_id | INT (PK) | Surrogate primary key assigned at data creation | 1, 42, 1000 |
| Timestamp | DATETIME | Date and time the incident was recorded | 1/1/18 0:00 |
| Incident_Type | VARCHAR (categorical) | Type of medical emergency | Cardiac Arrest, Accident, Other |
| Incident_Severity | VARCHAR (categorical) | Severity level of the incident | Low, Medium, High |
| Emergency_Level | VARCHAR (categorical) | Operational urgency classification | Minor, Major, Critical |
| Region_Type | VARCHAR (categorical) | Geographic classification of the incident location | Urban, Suburban, Rural |
| Road_Type | VARCHAR (categorical) | Type of road at or near the incident | Highway, Local Road, Unpaved Road |
| Number_of_Injuries | INT | Number of people injured in the incident | 1, 2, 3 |
| Distance_to_Incident | FLOAT | Distance from dispatch origin to incident in km | 8.89, 28.10, 40.43 |
| Response_Time | FLOAT | Time in minutes from dispatch to arrival | 5.0, 18.77, 24.49 |
| Label | VARCHAR (categorical) | Recorded dispatch outcome; classification target | Ambulance Only, Drone Only, Hybrid Dispatch |

`environmental_conditions`

| Feature Name | Data Type | Description | Example |
|---|---|---|---|
| condition_id | INT (PK) | Surrogate primary key | 1, 42, 1000 |
| incident_id | INT (FK) | Foreign key linking to incident table | 1, 42, 1000 |
| Weather_Condition | VARCHAR (categorical) | Weather at time of incident | Clear, Rainy, Stormy |
| Weather_Impact | VARCHAR (categorical) | Operational impact of weather on response | None, Moderate, Severe |
| Traffic_Congestion | VARCHAR (categorical) | Road congestion level at time of incident | Low, Moderate, High |
| Air_Traffic | VARCHAR (categorical) | Aerial congestion level relevant to drone routing | Low, Medium, High |

`resource`

| Feature Name | Data Type | Description | Example |
|---|---|---|---|
| resource_id | INT (PK) | Surrogate primary key | 1, 42, 1000 |
| incident_id | INT (FK) | Foreign key linking to incident table | 1, 42, 1000 |
| Drone_Availability | VARCHAR (categorical) | Whether a drone was available at dispatch time | Available, Unavailable |
| Ambulance_Availability | VARCHAR (categorical) | Whether an ambulance was available at dispatch time | Available, Unavailable |
| Battery_Life | FLOAT | Drone battery charge percentage at dispatch | 64.82, 78.49, 84.19 |
| Drone_Speed | FLOAT | Drone speed in km/h at time of dispatch | 45.90, 63.72, 74.58 |
| Ambulance_Speed | FLOAT | Ambulance speed in km/h at time of dispatch | 25.04, 43.55, 49.39 |
| Payload_Weight | FLOAT | Weight of medical payload carried by drone in kg | 0.80, 5.76, 9.47 |
| Fuel_Level | FLOAT | Ambulance fuel level as a percentage | 60.23, 84.08, 97.72 |

`dispatch`

| Feature Name | Data Type | Description | Example |
|---|---|---|---|
| dispatch_id | INT (PK) | Surrogate primary key | 1, 42, 1000 |
| incident_id | INT (FK) | Foreign key linking to incident table | 1, 42, 1000 |
| Dispatch_Coordinator | VARCHAR (categorical) | Whether dispatch decision was made by AI or a human | AI, Human |
| Specialist_Availability | VARCHAR (categorical) | Whether a medical specialist was available at the hospital | Available, Unavailable |
| Hospital_Capacity | INT | Percentage of hospital capacity available at time of incident | 15, 52, 94 |

### Uncertainty Quantification 
All numerical statistics are approximate, derived from the sample data provided. Full dataset statistics should be recomputed after loading all parquet files.

| Feature | Table | Type | Description | Uncertainty Notes |
|---|---|---|---|---|
| Response_Time | incident | FLOAT | Minutes from dispatch to arrival | Primary regression target. A minimum value of 5.0 minutes appears across multiple records regardless of distance or conditions, suggesting possible data floor or clipping in the synthetic generation process. High variance expected across urban vs. rural incidents. |
| Distance_to_Incident | incident | FLOAT | km from dispatch to incident | Range observed from ~8 to ~40 km in sample. Synthetic distances may not reflect Sydney's actual road network geometry; straight-line distance likely used rather than routed distance, introducing underestimation uncertainty. |
| Number_of_Injuries | incident | INT | Count of injured persons | Low range (1–3 in sample). As a synthetically generated feature, the distribution may not reflect real-world injury count patterns, which tend to be heavily right-skewed for major incidents. |
| Battery_Life | resource | FLOAT | Drone battery % at dispatch | Range ~64–85% in sample. Uncertainty arises from the fact that battery depletion during flight is not modeled; the recorded value is a snapshot at dispatch, not accounting for round-trip energy cost. |
| Drone_Speed | resource | FLOAT | Drone speed in km/h | Range ~46–75 km/h in sample. Does not account for wind speed, payload weight effects on speed, or air traffic constraints, all of which would reduce realized speed in practice. |
| Ambulance_Speed | resource | FLOAT | Ambulance speed in km/h | Range ~25–49 km/h in sample. Does not encode route-specific constraints such as road type or real-time congestion changes mid-transit. |
| Payload_Weight | resource | FLOAT | Medical payload in kg | Range ~0.8–9.5 kg in sample. The effect of payload weight on drone range and speed is not explicitly modeled in the dataset, introducing uncertainty when using this feature in routing optimization. |
| Fuel_Level | resource | FLOAT | Ambulance fuel % at dispatch | Range ~60–98% in sample. Fuel consumption during transit is not tracked, so this represents dispatch-time state only. |
| Hospital_Capacity | dispatch | INT | % hospital capacity available | Range 15–94% in sample. This is a snapshot value at incident time; capacity can change significantly during a response, meaning the recorded value may not reflect actual availability at patient arrival. |
