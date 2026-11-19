# Key Findings & Recommendations

**Overall:** 41.1% of 5,992 orders were delivered late
(avg 2.69 days late). INR 8,947,541 of order value
(40.9%) was tied to delayed shipments.

## Findings
1. **Carrier matters most.** QuickKart Logistics has a 60.7% delay rate versus
   29.2% for SwiftShip; differences are statistically significant (chi-square p < 0.001).
2. **Seasonality.** Peak delay month is month 10 (52.0%).
   Festive season delay rate: 49.8% vs
   Winter 37.5%.
3. **Distance.** National (>1000 km) shipments are late 37.3% of the time versus
   32.1% for local ones.
4. **Region.** Worst region: East (50.3%); best: South (38.5%).
5. **Root causes.** Carrier Capacity and Warehouse Dispatch Delay are the top reasons; the top 3 causes explain
   58% of all delays (Pareto).
6. **Customer impact.** Average rating drops from 4.49 (on time) to
   3.45 (delayed).
7. **Model.** A logistic regression predicts delay risk at order time with ROC-AUC = 0.69.

## Recommendations
- Re-allocate volume away from QuickKart Logistics on long-distance and East-region lanes, or renegotiate SLAs / penalties.
- Add buffer capacity and pre-position inventory before the monsoon (Jul-Sep) and festive (Oct-Nov) periods.
- Promise realistic delivery dates for >1000 km lanes (add +1 day) to cut "late" perception.
- Fix warehouse dispatch backlog and stock-out issues (controllable, internal causes).
- Use the delay-risk model to flag high-risk orders for proactive customer communication.
