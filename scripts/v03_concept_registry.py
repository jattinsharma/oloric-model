#!/usr/bin/env python3
"""
OLORIC v0.3 Dataset Generator

Addresses four root causes from the v0.2 audit:
  1. Task-type following collapse (concept-specific conversations, task-aligned targets)
  2. Cross-domain example bleed (per-concept example mappings)
  3. Misconception correction incoherence (concept-misconception pairing)
  4. Document grounding failure (real evidence anchoring)

Generates 480 records (20 categories × 24 records each), 6 domains, 10 concepts/domain.
Deterministic with seed=42.
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# CONSTANTS
# ============================================================

SEED = 42
TOTAL_RECORDS = 480
RECORDS_PER_CATEGORY = 24
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

VALID_CATEGORIES = [
    "simple_explanation", "simplification", "analogy", "concrete_example",
    "numerical_example", "prerequisite_detection", "misconception_detection",
    "follow_up_questions", "multi_turn_tutoring", "repeated_confusion",
    "strategy_switching", "diagnostic_questions", "hint_based_teaching",
    "practice_questions", "error_correction", "partial_understanding",
    "understanding_confirmation", "memory_generation", "document_grounded",
    "context_retention",
]

VALID_DOMAINS = [
    "economics", "accountancy", "mathematics",
    "science", "nutrition_food_science", "general_academic",
]

# ============================================================
# PER-CONCEPT CONTENT REGISTRY
# ============================================================
# Each concept has: explanation, reason, concrete_examples, numerical_examples,
# misconceptions (with refutations), analogies, prerequisites, document_evidence,
# formulas, practice_problems, hints, follow_up_questions, diagnostic_questions

CONCEPT_REGISTRY: Dict[str, Dict[str, Any]] = {}

def _reg(domain: str, concept: str, **kwargs):
    """Register a concept with all its content."""
    CONCEPT_REGISTRY[(domain, concept)] = {
        "domain": domain,
        "concept": concept,
        **kwargs,
    }

# --- ECONOMICS ---

_reg("economics", "MPC (Marginal Propensity to Consume)",
    explanation="the fraction of additional income that a household spends on consumption rather than saving",
    reason="it determines the size of the spending multiplier and helps predict how fiscal policy stimulates the economy",
    concrete_examples=[
        "If a family receives a $500 tax rebate and spends $400 of it on groceries and clothing, their MPC is 0.80.",
        "When workers get a $200 bonus and save $60 while spending $140, the MPC is 0.70.",
        "A retiree who spends every dollar of a $300 pension increase has an MPC of 1.0.",
    ],
    numerical_examples=[
        "If income rises from $40,000 to $42,000 and consumption rises from $35,000 to $36,600, MPC = $1,600/$2,000 = 0.80.",
        "With MPC = 0.75, the spending multiplier is 1/(1−0.75) = 4, so a $100 billion government injection raises GDP by $400 billion.",
        "If MPC = 0.90, each additional $1 of income generates $0.90 of consumption, leaving only $0.10 for saving.",
    ],
    misconceptions=[
        ("MPC is always the same for rich and poor households",
         "Empirically, lower-income households tend to have higher MPCs because a larger share of additional income goes to necessities, while wealthier households save more of their marginal income."),
    ],
    analogies=["MPC is like a leaky bucket: pour income in at the top and some leaks out as saving; the bigger the holes, the lower the MPC."],
    prerequisites=["basic algebra", "percentage calculations"],
    formulas=["MPC = ΔC/ΔY", "Multiplier = 1/(1−MPC)"],
    document_evidence=[
        "According to the textbook (p.45): 'The MPC measures the proportion of each additional dollar of income that is devoted to consumption expenditure.'",
        "Table 3.2 shows that households with income below $30,000 have an average MPC of 0.92, while those above $100,000 average 0.65.",
    ],
    practice_problems=[
        "If a household's income increases by $5,000 and consumption increases by $3,500, calculate the MPC and the spending multiplier.",
        "Given MPC = 0.6, how much total GDP increase results from a $200 billion fiscal stimulus?",
    ],
    hints=[
        "Remember that MPC + MPS = 1, where MPS is the marginal propensity to save.",
        "The multiplier formula uses MPC in the denominator: 1/(1−MPC).",
    ],
    follow_up_questions=[
        "What happens to the multiplier effect if households suddenly start saving more?",
        "How would a change in MPC from 0.8 to 0.6 affect the government's ability to stimulate the economy?",
    ],
    diagnostic_questions=[
        "Can you explain what happens to each additional dollar of income?",
        "If someone tells you their MPC is 1.2, what would you say is wrong with that claim?",
    ],
)

_reg("economics", "GDP",
    explanation="the total monetary value of all final goods and services produced within a country's borders in a specific time period",
    reason="it is the primary indicator of a country's economic health and standard of living",
    concrete_examples=[
        "If a country produces 1 million cars worth $20,000 each and 500,000 tons of wheat worth $200 each, those values contribute to GDP.",
        "A haircut costing $30 counts toward GDP because it is a final service produced domestically.",
        "An imported smartphone does NOT count toward the importing country's GDP because it was produced abroad.",
    ],
    numerical_examples=[
        "If C=$10T, I=$3T, G=$4T, and net exports (X−M)=−$0.5T, then GDP = $10T + $3T + $4T − $0.5T = $16.5 trillion.",
        "If nominal GDP is $20 trillion and the GDP deflator is 120, real GDP = $20T/1.20 = $16.67 trillion.",
        "A country's GDP growth from $15T to $15.45T represents a 3% annual growth rate.",
    ],
    misconceptions=[
        ("GDP measures the total well-being of a country's citizens",
         "GDP only measures market output. It excludes unpaid household work, leisure, environmental quality, income distribution, and other factors that affect well-being. A country can have high GDP but extreme inequality."),
    ],
    analogies=["GDP is like a cash register total for an entire country — it tallies every final sale but doesn't tell you who's buying or how happy they are."],
    prerequisites=["basic arithmetic", "graph interpretation"],
    formulas=["GDP = C + I + G + (X−M)"],
    document_evidence=[
        "The chapter states: 'GDP equals the sum of consumption, investment, government spending, and net exports.'",
        "Figure 2.1 illustrates that consumer spending accounts for roughly 68% of U.S. GDP.",
    ],
    practice_problems=[
        "Given C=$8T, I=$2T, G=$3T, X=$1.5T, M=$2T, calculate GDP.",
        "If nominal GDP rises 5% but inflation is 3%, estimate real GDP growth.",
    ],
    hints=[
        "Remember the expenditure approach: add up what everyone spent.",
        "Only count FINAL goods — don't double-count intermediate inputs.",
    ],
    follow_up_questions=[
        "Why do we subtract imports when calculating GDP?",
        "What's the difference between nominal and real GDP, and why does it matter?",
    ],
    diagnostic_questions=[
        "If a bakery buys $500 of flour and sells $2,000 of bread, how much does each transaction contribute to GDP?",
        "Does a used car sale count toward GDP? Why or why not?",
    ],
)

_reg("economics", "Inflation",
    explanation="a sustained increase in the general price level of goods and services over time, reducing the purchasing power of money",
    reason="it affects interest rates, wages, savings, and investment decisions throughout the economy",
    concrete_examples=[
        "If a loaf of bread costs $2.50 this year but cost $2.00 last year, the price of bread inflated by 25%.",
        "When rent increases from $1,200/month to $1,260/month, the landlord is reflecting a 5% inflation in housing costs.",
        "A car that cost $25,000 five years ago now costs $30,000, showing cumulative inflation in auto prices.",
    ],
    numerical_examples=[
        "If the CPI rises from 250 to 260, the inflation rate = (260−250)/250 × 100 = 4%.",
        "With 3% annual inflation, $100 today has the purchasing power of about $97.09 next year.",
        "If nominal interest is 6% and inflation is 2%, the real interest rate is approximately 4%.",
    ],
    misconceptions=[
        ("Inflation means every single price goes up at the same rate",
         "Inflation measures the AVERAGE price level change. Individual prices can rise, fall, or stay constant even during inflation. Technology prices often fall while food prices rise."),
    ],
    analogies=["Inflation is like a slow leak in a tire: your money gradually loses its 'pressure' (purchasing power), and you need to keep pumping in more income just to stay at the same level."],
    prerequisites=["percentage calculations", "graph interpretation"],
    formulas=["Inflation rate = (CPI_new − CPI_old)/CPI_old × 100"],
    document_evidence=[
        "The text explains: 'Inflation erodes purchasing power, meaning each dollar buys fewer goods over time.'",
        "Historical data (Table 4.1) shows U.S. inflation averaged 2.1% annually from 2000–2019.",
    ],
    practice_problems=[
        "If the CPI was 245 in January and 252 in December, what was the annual inflation rate?",
        "Calculate the real return on a savings account paying 4% interest when inflation is 3%.",
    ],
    hints=[
        "The CPI (Consumer Price Index) is the most common measure — it tracks a basket of goods.",
        "Real values = Nominal values adjusted for inflation.",
    ],
    follow_up_questions=[
        "How does unexpected inflation redistribute wealth between borrowers and lenders?",
        "What tools does a central bank use to control inflation?",
    ],
    diagnostic_questions=[
        "If your salary increases by 3% but inflation is 4%, are you better off or worse off?",
        "Can you name a time when prices of some goods fell even though overall inflation was positive?",
    ],
)

_reg("economics", "Supply and Demand",
    explanation="the model describing how the price and quantity of a good are determined by the interaction of buyers (demand) and sellers (supply) in a market",
    reason="it is the foundational framework for understanding how markets allocate resources and set prices",
    concrete_examples=[
        "When a drought reduces wheat supply, the price of bread rises because the same number of buyers compete for less bread.",
        "When a new smartphone model launches and demand surges, stores raise prices until demand and supply balance.",
        "When avocado imports increase (supply shifts right), the price of avocados falls at grocery stores.",
    ],
    numerical_examples=[
        "If the demand function is Qd = 100 − 2P and supply is Qs = 20 + 3P, equilibrium occurs where 100 − 2P = 20 + 3P → P = $16, Q = 68 units.",
        "A 20% increase in supply with constant demand typically leads to a price decrease, the magnitude depending on demand elasticity.",
        "If price rises from $5 to $6 and quantity demanded drops from 200 to 170, the price elasticity of demand is −(30/200)/(1/5) = −0.75.",
    ],
    misconceptions=[
        ("Higher prices always reduce the quantity demanded of every good",
         "Giffen goods and Veblen goods are exceptions. Giffen goods see increased demand when price rises because the income effect dominates. Veblen goods are demanded more at higher prices because of their status signaling value."),
    ],
    analogies=["Supply and demand is like a tug-of-war between buyers and sellers — the equilibrium price is where neither side can pull the rope any further."],
    prerequisites=["graph interpretation", "basic algebra"],
    formulas=["At equilibrium: Qd = Qs"],
    document_evidence=[
        "The textbook states: 'Market equilibrium occurs at the price where quantity demanded equals quantity supplied.'",
        "Figure 3.4 shows the classic X-shaped supply-demand diagram with the equilibrium at their intersection.",
    ],
    practice_problems=[
        "Given Qd = 50 − P and Qs = −10 + 2P, find the equilibrium price and quantity.",
        "If a government sets a price ceiling of $4 in the above market, what happens to the quantity supplied and demanded?",
    ],
    hints=[
        "Set Qd = Qs and solve for P to find equilibrium.",
        "Price ceilings below equilibrium cause shortages; price floors above equilibrium cause surpluses.",
    ],
    follow_up_questions=[
        "What shifts the demand curve to the right, and how does that affect equilibrium?",
        "How does the concept of elasticity modify our prediction about price changes?",
    ],
    diagnostic_questions=[
        "Can you sketch a supply and demand diagram and label the equilibrium?",
        "What is the difference between a change in demand and a change in quantity demanded?",
    ],
)

_reg("economics", "Opportunity Cost",
    explanation="the value of the next best alternative forgone when making a choice",
    reason="it forces decision-makers to consider hidden costs and leads to more efficient resource allocation",
    concrete_examples=[
        "If you spend $50 on a concert ticket, the opportunity cost is whatever else you would have done with that $50 — perhaps buying textbooks or saving it.",
        "A farmer who uses a field to grow corn gives up the revenue from growing soybeans on that same field.",
        "A student who studies for 4 hours gives up 4 hours of paid work, leisure, or sleep.",
    ],
    numerical_examples=[
        "If a lawyer earning $200/hour spends 3 hours mowing the lawn instead of working, the opportunity cost is $600 in lost billable hours.",
        "A company investing $1 million in Project A (expected return 8%) instead of Project B (expected return 12%) incurs an opportunity cost of 4% × $1M = $40,000.",
        "Attending a 4-year college with tuition of $40,000/year has an opportunity cost that includes $30,000/year in forgone wages, totaling $280,000.",
    ],
    misconceptions=[
        ("Opportunity cost only includes monetary costs",
         "Opportunity cost includes ALL forgone benefits — time, enjoyment, health, and other non-monetary values. A free concert still has an opportunity cost: the time you could have spent studying."),
    ],
    analogies=["Opportunity cost is like choosing a path at a fork in the road — you can't travel both, so the scenic route you didn't take is your opportunity cost."],
    prerequisites=["basic arithmetic"],
    formulas=["Opportunity Cost = Return of Next Best Alternative − Return of Chosen Option"],
    document_evidence=[
        "The text defines: 'Every choice involves a trade-off; the opportunity cost is the value of the road not taken.'",
        "Example 5.2 illustrates that even free goods have opportunity costs when time is scarce.",
    ],
    practice_problems=[
        "You have $10,000 to invest. Option A returns 5%, Option B returns 7%. If you choose A, what is the opportunity cost?",
        "If a nurse earning $40/hour spends 2 hours cooking instead of ordering $15 takeout, calculate the total cost of cooking including opportunity cost.",
    ],
    hints=[
        "Always ask: 'What am I giving up by making this choice?'",
        "Remember that opportunity cost is about the NEXT BEST alternative, not all alternatives.",
    ],
    follow_up_questions=[
        "Why is it important to consider opportunity cost even when something appears free?",
        "How does opportunity cost guide a country's decision about which goods to produce?",
    ],
    diagnostic_questions=[
        "If a student can either work a $15/hr job or study, what is the opportunity cost of studying for 3 hours?",
        "Can something with a zero monetary price still have an opportunity cost? Explain.",
    ],
)

_reg("economics", "Elasticity",
    explanation="a measure of how responsive the quantity demanded or supplied of a good is to a change in price, income, or other factors",
    reason="it helps businesses set prices, governments predict tax revenue impacts, and economists understand market sensitivity",
    concrete_examples=[
        "Gasoline has low price elasticity: even when prices rise 30%, people reduce driving only slightly because they need fuel for commuting.",
        "Luxury watches are highly elastic: a 10% price increase can cause demand to drop by 30% or more.",
        "Insulin is perfectly inelastic for diabetics: regardless of price changes, they must purchase the same amount.",
    ],
    numerical_examples=[
        "If a 10% price increase causes a 20% decrease in quantity demanded, price elasticity = −20%/10% = −2.0 (elastic).",
        "If bread price rises 15% and quantity demanded falls only 3%, elasticity = −3%/15% = −0.2 (inelastic).",
        "Total revenue test: if price rises and total revenue falls, demand is elastic; if revenue rises, demand is inelastic.",
    ],
    misconceptions=[
        ("Elastic demand means the good is a luxury",
         "Elasticity depends on availability of substitutes, time horizon, and budget share — not just whether a good is a luxury. Salt is cheap and necessary but inelastic; restaurant meals are not luxury goods but have many substitutes, making them elastic."),
    ],
    analogies=["Elasticity is like a rubber band — an elastic good stretches a lot (big quantity change) with a small pull (price change), while an inelastic good is stiff and barely moves."],
    prerequisites=["percentage calculations", "basic algebra"],
    formulas=["Price Elasticity of Demand = %ΔQd / %ΔP"],
    document_evidence=[
        "Section 4.3 states: 'Goods with many close substitutes tend to have elastic demand.'",
        "Table 4.2 lists estimated elasticities: gasoline (−0.3), airline travel (−1.1), restaurant meals (−1.6).",
    ],
    practice_problems=[
        "If the price of coffee rises from $4 to $5 and sales drop from 1,000 to 800 cups, calculate the price elasticity of demand.",
        "A firm raises price by 5% and sees revenue increase. Is demand elastic or inelastic? Explain.",
    ],
    hints=[
        "Elasticity greater than 1 (in absolute value) means elastic; less than 1 means inelastic.",
        "More substitutes → more elastic. Longer time to adjust → more elastic.",
    ],
    follow_up_questions=[
        "Why does the time horizon matter for elasticity?",
        "How should a firm set its price if it knows demand is elastic?",
    ],
    diagnostic_questions=[
        "If a 5% price cut leads to a 2% increase in quantity demanded, is this good elastic or inelastic?",
        "Why might the elasticity of a good change over time?",
    ],
)

_reg("economics", "Comparative Advantage",
    explanation="the ability of a country or individual to produce a good at a lower opportunity cost than another, even if they are less efficient in absolute terms",
    reason="it explains why trade benefits all parties and is the foundation of international trade theory",
    concrete_examples=[
        "Even if a doctor is faster at typing than a secretary, the doctor has a comparative advantage in medicine because the opportunity cost of typing (lost medical fees) is very high.",
        "Brazil has a comparative advantage in coffee production because its climate and soil make the opportunity cost of growing coffee lower than for most other countries.",
        "China may have an absolute advantage in both shirts and computers, but if its opportunity cost of shirts is lower, it has a comparative advantage in shirts.",
    ],
    numerical_examples=[
        "Country A produces 1 car in 10 hours or 5 tons of wheat in 10 hours. Country B produces 1 car in 20 hours or 2 tons of wheat in 20 hours. Country A's opportunity cost of 1 car = 5 tons of wheat; Country B's = 2 tons. Country B has comparative advantage in cars.",
        "If the US can produce 100 units of tech or 50 units of textiles, and Vietnam can produce 20 units of tech or 40 units of textiles, Vietnam has a comparative advantage in textiles (OC = 0.5 tech per textile vs US's 2).",
        "With trade, if both countries specialize, total output increases even though one country is absolutely better at everything.",
    ],
    misconceptions=[
        ("A country that is worse at producing everything has no basis for trade",
         "Comparative advantage shows that even a country less efficient at everything can still benefit from trade by specializing in goods where its opportunity cost is lowest. This is different from absolute advantage."),
    ],
    analogies=["Comparative advantage is like a basketball player who is also a great cook — it makes more sense for them to play basketball (where their edge is greatest) and hire a chef, even if they can cook well."],
    prerequisites=["basic arithmetic", "opportunity cost"],
    formulas=["Opportunity Cost of Good X = Amount of Good Y sacrificed / Amount of Good X produced"],
    document_evidence=[
        "David Ricardo's principle (p.78): 'Trade benefits both parties when each specializes in the good where their opportunity cost is lowest.'",
        "Table 6.1 compares production possibilities for two countries to illustrate comparative advantage.",
    ],
    practice_problems=[
        "Country X produces 30 bushels of corn or 10 yards of cloth per day. Country Y produces 20 bushels of corn or 15 yards of cloth. Which country has a comparative advantage in cloth?",
        "Explain why both countries benefit from trade, even if Country X is better at producing both goods.",
    ],
    hints=[
        "Calculate the opportunity cost for each good in each country, then compare.",
        "Comparative advantage is about RELATIVE efficiency, not absolute efficiency.",
    ],
    follow_up_questions=[
        "How does comparative advantage change when technology improves in one country?",
        "Can a country have a comparative advantage in everything? Why or why not?",
    ],
    diagnostic_questions=[
        "What is the difference between absolute advantage and comparative advantage?",
        "If you're the best at everything in a group project, should you do all the work? How does this relate to comparative advantage?",
    ],
)

_reg("economics", "Fiscal Policy",
    explanation="the use of government spending and taxation to influence the economy's aggregate demand, employment, and price level",
    reason="it allows governments to stabilize business cycles, combat recessions, and manage public services",
    concrete_examples=[
        "During the 2008 recession, the US government passed a stimulus package increasing spending on infrastructure to boost employment and demand.",
        "A government cutting income tax rates puts more money in consumers' pockets, encouraging spending and economic growth.",
        "Increasing military spending is expansionary fiscal policy that raises aggregate demand.",
    ],
    numerical_examples=[
        "If the government increases spending by $100 billion and MPC = 0.8, the multiplier is 5, so GDP increases by $500 billion.",
        "A $50 billion tax cut with MPC = 0.75 has a tax multiplier of −MPC/(1−MPC) = −3, increasing GDP by $150 billion.",
        "If the budget deficit is $800 billion (spending $4.5T, revenue $3.7T), the government must borrow the difference.",
    ],
    misconceptions=[
        ("Government spending always leads to economic growth",
         "If the economy is already at full employment, additional government spending can cause inflation ('crowding out') rather than real growth, and the borrowed money raises interest rates, reducing private investment."),
    ],
    analogies=["Fiscal policy is like a thermostat for the economy: the government turns up spending (heat) during recessions and turns it down (cool) during booms to maintain a comfortable temperature."],
    prerequisites=["basic algebra", "graph interpretation"],
    formulas=["Government Spending Multiplier = 1/(1−MPC)", "Tax Multiplier = −MPC/(1−MPC)"],
    document_evidence=[
        "The chapter notes: 'Fiscal policy is most effective when the economy operates below full employment and monetary policy alone is insufficient.'",
        "Case study 7.3 examines the 2009 American Recovery and Reinvestment Act's impact on employment.",
    ],
    practice_problems=[
        "With MPC = 0.80, compare the GDP impact of a $200 billion spending increase vs. a $200 billion tax cut.",
        "Explain why the spending multiplier is larger than the tax multiplier.",
    ],
    hints=[
        "Expansionary = more spending or less taxes. Contractionary = less spending or more taxes.",
        "The spending multiplier is always larger than the tax multiplier because some of the tax cut is saved.",
    ],
    follow_up_questions=[
        "What is the difference between automatic stabilizers and discretionary fiscal policy?",
        "Why might there be a time lag between implementing fiscal policy and seeing its effects?",
    ],
    diagnostic_questions=[
        "During a recession, should the government increase or decrease spending? Why?",
        "What is 'crowding out' and how does it limit the effectiveness of fiscal policy?",
    ],
)

_reg("economics", "Monetary Policy",
    explanation="the actions taken by a central bank to manage the money supply and interest rates to achieve macroeconomic objectives like stable prices, full employment, and economic growth",
    reason="it is the primary tool for controlling inflation and stabilizing the economy in the short to medium term",
    concrete_examples=[
        "When the Federal Reserve lowers the federal funds rate, borrowing becomes cheaper, encouraging businesses to invest and consumers to spend.",
        "During high inflation, the central bank raises interest rates to make borrowing expensive, cooling demand and slowing price increases.",
        "Quantitative easing involves the central bank buying government bonds to inject money into the economy.",
    ],
    numerical_examples=[
        "If the central bank lowers the interest rate from 5% to 3%, a business considering a $1 million loan saves $20,000/year in interest, making the investment more attractive.",
        "With a reserve requirement of 10%, a $1,000 deposit can create up to $10,000 in new money through the money multiplier (1/0.10).",
        "If the target inflation rate is 2% and actual inflation is 4%, the Taylor Rule suggests the central bank should raise rates.",
    ],
    misconceptions=[
        ("The central bank directly controls all interest rates in the economy",
         "The central bank sets the overnight rate (e.g., federal funds rate) at which banks lend to each other. Other rates (mortgages, car loans, corporate bonds) are influenced by this rate but are also determined by risk, term, and market expectations."),
    ],
    analogies=["Monetary policy is like controlling the water pressure in a plumbing system — the central bank adjusts the main valve (interest rates) to increase or decrease the flow of money through the economy."],
    prerequisites=["percentage calculations", "basic algebra"],
    formulas=["Money Multiplier = 1/Reserve Requirement"],
    document_evidence=[
        "The text states: 'By adjusting the federal funds rate, the Federal Reserve influences borrowing costs throughout the economy.'",
        "Figure 8.2 shows the inverse relationship between interest rates and investment spending.",
    ],
    practice_problems=[
        "If the reserve requirement is 20%, what is the money multiplier? How much total money can $5,000 in new deposits create?",
        "Explain the transmission mechanism: how does lowering the fed funds rate eventually boost GDP?",
    ],
    hints=[
        "Lower interest rates → cheaper borrowing → more investment and consumption → higher GDP.",
        "The money multiplier is 1 divided by the reserve requirement ratio.",
    ],
    follow_up_questions=[
        "What happens when interest rates are already near zero — what tools does the central bank still have?",
        "How do inflation expectations affect the central bank's ability to manage the economy?",
    ],
    diagnostic_questions=[
        "If the economy is overheating, should the central bank raise or lower interest rates?",
        "What is the difference between fiscal policy and monetary policy?",
    ],
)

_reg("economics", "Externalities",
    explanation="costs or benefits of a market transaction that affect third parties who are not directly involved in the transaction",
    reason="they cause market failure because private costs/benefits diverge from social costs/benefits, justifying government intervention",
    concrete_examples=[
        "A factory polluting a river imposes a negative externality on downstream fishermen and communities who bear health and cleanup costs.",
        "Getting vaccinated creates a positive externality because it protects not just you but also those around you through herd immunity.",
        "A beekeeper's bees pollinate a neighboring farmer's crops — a positive externality the beekeeper is not compensated for.",
    ],
    numerical_examples=[
        "If producing one ton of steel costs $100 privately but causes $30 in pollution damage, the social cost is $130. Without intervention, the market overproduces steel.",
        "A Pigouvian tax of $30/ton on steel production would internalize the externality, aligning private and social costs.",
        "If each vaccination costs $20 but provides $50 in community health benefits, the social benefit ($70) exceeds the private benefit ($20).",
    ],
    misconceptions=[
        ("Externalities only refer to pollution and environmental damage",
         "Externalities include ANY spillover — positive (education, vaccination, R&D) and negative (noise, congestion, secondhand smoke). They can be production or consumption externalities in any sector."),
    ],
    analogies=["An externality is like secondhand smoke — your smoking affects the health of everyone nearby, and they didn't choose to breathe it in."],
    prerequisites=["basic algebra", "graph interpretation"],
    formulas=["Social Cost = Private Cost + External Cost", "Social Benefit = Private Benefit + External Benefit"],
    document_evidence=[
        "The text explains: 'When external costs exist, the market equilibrium quantity exceeds the socially optimal quantity.'",
        "Diagram 9.1 shows the deadweight loss triangle caused by overproduction with a negative externality.",
    ],
    practice_problems=[
        "If the private cost of producing a widget is $10 and the external cost is $3, what Pigouvian tax should the government impose?",
        "Draw a supply-demand diagram showing how a negative externality leads to overproduction and label the deadweight loss.",
    ],
    hints=[
        "Negative externalities → social cost > private cost → overproduction.",
        "Positive externalities → social benefit > private benefit → underproduction.",
    ],
    follow_up_questions=[
        "What is the Coase theorem, and when does it suggest government intervention is unnecessary?",
        "How do cap-and-trade systems address negative externalities differently from taxes?",
    ],
    diagnostic_questions=[
        "Can you give an example of a positive externality in education?",
        "Why does the free market produce too much of a good with negative externalities?",
    ],
)

# --- ACCOUNTANCY ---

_reg("accountancy", "Double-Entry Bookkeeping",
    explanation="an accounting system where every financial transaction is recorded in at least two accounts — a debit in one and a credit in another — keeping the accounting equation balanced",
    reason="it provides a complete record of transactions, enables error detection, and ensures the accounting equation (Assets = Liabilities + Equity) always holds",
    concrete_examples=[
        "When a business buys a $5,000 computer with cash, it debits Equipment (asset increases) and credits Cash (asset decreases).",
        "When a company borrows $10,000 from a bank, it debits Cash (asset increases) and credits Loans Payable (liability increases).",
        "When a customer pays $1,000 for services, it debits Cash and credits Service Revenue.",
    ],
    numerical_examples=[
        "A company buys inventory for $3,000 on credit: Debit Inventory $3,000, Credit Accounts Payable $3,000. Assets and Liabilities both increase by $3,000.",
        "After recording Debit Rent Expense $2,000 and Credit Cash $2,000, the trial balance still balances because total debits = total credits.",
        "If total debits are $145,000 and total credits are $143,000, there is a $2,000 error that must be found.",
    ],
    misconceptions=[
        ("Debit always means an increase and credit always means a decrease",
         "The effect depends on the account type. Debits increase assets and expenses but decrease liabilities, equity, and revenue. Credits do the opposite. For example, crediting Cash decreases it (asset), but crediting Revenue increases it."),
    ],
    analogies=["Double-entry bookkeeping is like a balanced see-saw — every time you add weight (debit) on one side, you must add equal weight (credit) on the other to keep it level."],
    prerequisites=["basic arithmetic"],
    formulas=["Assets = Liabilities + Equity", "Debits must always equal Credits"],
    document_evidence=[
        "Chapter 2 states: 'The fundamental principle of double-entry is that every transaction affects at least two accounts, maintaining the balance of the accounting equation.'",
        "Example 2.3 traces a sale on credit through both the revenue and receivable accounts.",
    ],
    practice_problems=[
        "Record the journal entry for purchasing office supplies worth $500 with cash.",
        "A company receives $8,000 from a customer who owed money. Write the journal entry and explain which accounts are affected.",
    ],
    hints=[
        "Ask: What comes in? What goes out? Debit what comes in, credit what goes out.",
        "Remember DEALER: Dividends, Expenses, Assets increase with Debits; Liabilities, Equity, Revenue increase with Credits.",
    ],
    follow_up_questions=[
        "What happens if a transaction is only recorded in one account?",
        "How does double-entry help detect errors?",
    ],
    diagnostic_questions=[
        "If you debit Cash, what type of account must you credit to keep things balanced?",
        "Why is it called 'double-entry' and not 'single-entry'?",
    ],
)

_reg("accountancy", "Accrual Accounting",
    explanation="an accounting method that records revenues when earned and expenses when incurred, regardless of when cash is actually received or paid",
    reason="it provides a more accurate picture of a company's financial performance by matching revenues with the expenses that generated them in the same period",
    concrete_examples=[
        "A law firm records revenue in December when it completes a case, even though the client pays in January.",
        "Rent expense for December is recorded in December even if the check is mailed on January 2.",
        "Insurance paid annually ($12,000) is spread as $1,000/month expense, not recorded entirely in the payment month.",
    ],
    numerical_examples=[
        "A company delivers $50,000 of goods in March but receives payment in April. Under accrual accounting, March revenue = $50,000; cash accounting would show $0.",
        "Prepaid insurance of $6,000 for 6 months: each month records $1,000 in insurance expense through an adjusting entry.",
        "Salaries earned by employees in December ($15,000) but paid January 5 are accrued as a liability on December 31.",
    ],
    misconceptions=[
        ("Accrual accounting shows how much cash a company actually has",
         "Accrual accounting shows economic performance, not cash position. A company can report high profits under accrual accounting while having very little cash because revenue was recorded when earned, not when collected. The cash flow statement shows actual cash."),
    ],
    analogies=["Accrual accounting is like tracking calories you EAT, not when you buy the groceries — it measures economic activity when it happens, not when money changes hands."],
    prerequisites=["basic arithmetic", "double-entry bookkeeping basics"],
    formulas=["Revenue Recognition: Record when earned, not when cash received", "Matching Principle: Record expenses in the same period as related revenue"],
    document_evidence=[
        "The text states: 'Under GAAP, accrual accounting is required because it better reflects economic reality than cash-basis accounting.'",
        "Example 3.5 contrasts the same transactions under cash vs. accrual accounting to show the difference.",
    ],
    practice_problems=[
        "A consulting firm performs $20,000 of work in November but won't be paid until January. Record the November journal entry.",
        "Prepaid rent of $24,000 covers January–December. What adjusting entry is needed each month?",
    ],
    hints=[
        "Ask: When was the service performed or the benefit received? That's when you record it.",
        "Adjusting entries at period-end ensure revenues and expenses land in the correct period.",
    ],
    follow_up_questions=[
        "Why is the matching principle important for investors analyzing financial statements?",
        "What is the difference between a cash-basis and accrual-basis income statement?",
    ],
    diagnostic_questions=[
        "If a company received $10,000 cash for work it hasn't done yet, is that revenue under accrual accounting?",
        "Can a company be profitable on paper but unable to pay its bills? How is that possible?",
    ],
)

_reg("accountancy", "Financial Ratios",
    explanation="quantitative metrics computed from financial statement data that measure a company's profitability, liquidity, solvency, and efficiency",
    reason="they allow investors, creditors, and managers to assess financial health, compare companies, and make informed decisions",
    concrete_examples=[
        "A company with current assets of $200,000 and current liabilities of $100,000 has a current ratio of 2.0, indicating good short-term liquidity.",
        "A profit margin of 15% means that for every $1 of revenue, the company keeps $0.15 as profit.",
        "A debt-to-equity ratio of 1.5 means the company uses $1.50 of debt for every $1 of shareholder equity.",
    ],
    numerical_examples=[
        "Current Ratio = $300,000 / $150,000 = 2.0 — the company can cover short-term obligations twice over.",
        "ROE = Net Income ($50,000) / Shareholder Equity ($250,000) = 20%, meaning each dollar of equity generates $0.20 of profit.",
        "Inventory Turnover = COGS ($600,000) / Average Inventory ($100,000) = 6 times per year.",
    ],
    misconceptions=[
        ("A higher current ratio is always better",
         "An excessively high current ratio (e.g., 8.0) may indicate the company is inefficiently using its assets — sitting on too much cash or carrying excess inventory rather than investing for growth."),
    ],
    analogies=["Financial ratios are like a doctor's vital signs for a company — blood pressure (liquidity), heart rate (efficiency), and temperature (profitability) each tell part of the health story."],
    prerequisites=["basic arithmetic", "percentages"],
    formulas=["Current Ratio = Current Assets / Current Liabilities", "ROE = Net Income / Shareholder Equity", "Debt-to-Equity = Total Debt / Total Equity"],
    document_evidence=[
        "Table 5.1 presents industry-average financial ratios for benchmarking.",
        "The text advises: 'No single ratio tells the full story; always analyze multiple ratios together and compare to industry benchmarks.'",
    ],
    practice_problems=[
        "A company has net income of $80,000, total equity of $400,000, total assets of $1,000,000, and total debt of $600,000. Calculate ROE and debt-to-equity.",
        "If current assets are $120,000 and current liabilities are $80,000, what is the current ratio? What if the company uses $40,000 cash to pay a short-term loan?",
    ],
    hints=[
        "Liquidity ratios use current items from the balance sheet.",
        "Profitability ratios relate earnings to revenue or equity.",
    ],
    follow_up_questions=[
        "Why is comparing ratios across different industries potentially misleading?",
        "How does DuPont analysis decompose ROE into its component drivers?",
    ],
    diagnostic_questions=[
        "If a company's current ratio is 0.8, what does that tell you about its liquidity?",
        "What is the difference between ROE and ROA?",
    ],
)

_reg("accountancy", "Depreciation Methods",
    explanation="systematic approaches to allocating the cost of a tangible asset over its useful life, reflecting the consumption of economic benefit",
    reason="it matches the cost of assets to the revenue they help generate and provides accurate financial reporting of asset values",
    concrete_examples=[
        "A delivery truck purchased for $60,000 with a 5-year life and $10,000 salvage value depreciates $10,000/year under straight-line.",
        "A computer losing value quickly might use declining-balance depreciation, recording more expense in early years.",
        "A factory machine producing 100,000 units over its life uses units-of-production depreciation: expense per unit = (cost − salvage) / total units.",
    ],
    numerical_examples=[
        "Straight-line: ($50,000 cost − $5,000 salvage) / 10 years = $4,500/year depreciation expense.",
        "Double-declining balance, Year 1: $50,000 × (2/10) = $10,000. Year 2: ($50,000 − $10,000) × 0.20 = $8,000.",
        "Units-of-production: ($80,000 − $8,000) / 200,000 units = $0.36 per unit. If 30,000 units produced in Year 1, depreciation = $10,800.",
    ],
    misconceptions=[
        ("Depreciation represents the actual decline in market value of an asset",
         "Accounting depreciation is a cost allocation method, not a valuation technique. A building might appreciate in market value while still being depreciated on the books. Depreciation allocates historical cost to expense over time."),
    ],
    analogies=["Depreciation is like spreading the cost of a bus pass over all the months you use it, rather than recording the full cost when you first buy it."],
    prerequisites=["basic arithmetic", "percentages"],
    formulas=["Straight-Line = (Cost − Salvage) / Useful Life", "DDB Rate = (2 / Useful Life) × Book Value"],
    document_evidence=[
        "Section 8.2 states: 'Straight-line depreciation allocates equal amounts to each period, while accelerated methods front-load the expense.'",
        "Comparative Example 8.1 shows the same asset depreciated under three methods over 5 years.",
    ],
    practice_problems=[
        "Equipment costs $100,000, has a salvage value of $10,000, and a useful life of 6 years. Calculate Year 1 depreciation under straight-line and double-declining balance.",
        "A machine produced 50,000 units this year out of a total expected 200,000 units. Cost is $120,000, salvage $20,000. What is this year's depreciation?",
    ],
    hints=[
        "Straight-line gives constant annual expense; DDB gives higher early expenses.",
        "For units-of-production, the per-unit rate stays the same; the annual expense varies with output.",
    ],
    follow_up_questions=[
        "Why might a company choose accelerated depreciation over straight-line?",
        "How does the choice of depreciation method affect reported net income?",
    ],
    diagnostic_questions=[
        "What are the three pieces of information you need to calculate straight-line depreciation?",
        "If an asset's book value reaches its salvage value, can you continue to depreciate it?",
    ],
)

_reg("accountancy", "Cost-Volume-Profit Analysis",
    explanation="a managerial accounting method that examines how changes in costs, sales volume, and price affect a company's profit, particularly to find the break-even point",
    reason="it helps managers make pricing decisions, plan production levels, and assess the risk of different sales scenarios",
    concrete_examples=[
        "A bakery selling cupcakes at $5 each with variable cost of $2 and fixed costs of $3,000/month breaks even at 1,000 cupcakes.",
        "A software company with very low variable costs per user has a low break-even point but high contribution margin.",
        "If a restaurant's fixed costs rise due to rent increases, its break-even point shifts higher.",
    ],
    numerical_examples=[
        "Break-even units = Fixed Costs / Contribution Margin per unit = $10,000 / ($25 − $15) = 1,000 units.",
        "To earn a target profit of $5,000 with CM per unit of $10 and fixed costs of $10,000, required sales = ($10,000 + $5,000)/$10 = 1,500 units.",
        "Contribution margin ratio = ($25 − $15) / $25 = 40%. Break-even revenue = $10,000 / 0.40 = $25,000.",
    ],
    misconceptions=[
        ("The break-even point only applies to manufacturing companies",
         "CVP analysis applies to ANY business — services, retail, tech startups. Every business has fixed costs, variable costs, and revenue, so break-even analysis is universally applicable."),
    ],
    analogies=["Break-even is like filling a swimming pool — fixed costs are the initial volume you must fill, and each unit sold adds a bucket of water (contribution margin). Once the pool is full (break-even), every additional bucket overflows as profit."],
    prerequisites=["basic algebra", "percentages"],
    formulas=["Break-Even Units = Fixed Costs / (Price − Variable Cost)", "CM Ratio = (Price − VC) / Price"],
    document_evidence=[
        "The text explains: 'The contribution margin is the amount each unit contributes toward covering fixed costs and generating profit.'",
        "Graph 10.1 shows how total revenue and total cost lines intersect at the break-even point.",
    ],
    practice_problems=[
        "A company sells widgets at $40 each, variable cost is $24, fixed costs are $48,000. Find the break-even in units and dollars.",
        "What happens to the break-even point if variable cost increases by $4 per unit?",
    ],
    hints=[
        "Contribution margin per unit = Price − Variable cost. This is what each unit 'contributes' to covering fixed costs.",
        "After break-even, every unit sold adds exactly the contribution margin amount to profit.",
    ],
    follow_up_questions=[
        "How does operating leverage affect the riskiness of a business?",
        "What is the margin of safety and why is it important?",
    ],
    diagnostic_questions=[
        "What does it mean when a company is 'below break-even'?",
        "If you double fixed costs, what happens to the break-even point?",
    ],
)

_reg("accountancy", "Budget Variance Analysis",
    explanation="the process of comparing actual financial results to budgeted amounts to identify differences (variances) and understand their causes",
    reason="it helps managers evaluate performance, identify cost overruns or savings, and make corrective decisions",
    concrete_examples=[
        "If the marketing department budgeted $50,000 but spent $55,000, there is an unfavorable variance of $5,000.",
        "If actual sales revenue is $120,000 against a budget of $100,000, the $20,000 favorable variance indicates better-than-expected sales.",
        "A material price variance occurs when a company pays $12/unit for raw materials instead of the budgeted $10/unit.",
    ],
    numerical_examples=[
        "Sales volume variance: Budget = 10,000 units at $20 = $200,000. Actual = 11,000 units at $20 = $220,000. Favorable variance = $20,000.",
        "Labor efficiency variance: Standard 2 hrs/unit × 1,000 units = 2,000 hrs budgeted. Actual = 2,200 hrs. Unfavorable variance = 200 hrs × $15/hr = $3,000.",
        "Total variance = Price variance + Quantity variance. If price variance is −$2,000 (favorable) and quantity variance is +$3,500 (unfavorable), total is +$1,500 unfavorable.",
    ],
    misconceptions=[
        ("An unfavorable variance always means poor management",
         "Unfavorable variances can result from factors outside management's control — supply chain disruptions, raw material price spikes, or intentional quality improvements. Context matters more than the direction of the variance."),
    ],
    analogies=["Variance analysis is like comparing your actual driving route to GPS directions — deviations can be bad (traffic jams) or good (shortcuts), and you need to understand the reason to judge the outcome."],
    prerequisites=["basic arithmetic", "percentages"],
    formulas=["Variance = Actual − Budget", "Material Price Variance = (Actual Price − Standard Price) × Actual Quantity"],
    document_evidence=[
        "Chapter 12 states: 'Variance analysis decomposes the total difference between actual and budgeted results into price and quantity components.'",
        "Table 12.3 illustrates a complete variance analysis for a manufacturing department.",
    ],
    practice_problems=[
        "Budget: 5,000 units of raw material at $8/unit. Actual: 5,200 units at $7.50/unit. Calculate the material price variance and quantity variance.",
        "Is a $3,000 favorable labor rate variance always good news? Explain.",
    ],
    hints=[
        "Favorable = actual is better than budget (lower costs or higher revenue). Unfavorable = actual is worse.",
        "Decompose total variance into price/rate and quantity/efficiency components.",
    ],
    follow_up_questions=[
        "When should a manager investigate a variance versus ignoring it?",
        "How do flexible budgets improve variance analysis?",
    ],
    diagnostic_questions=[
        "If actual costs are lower than budgeted, is the variance favorable or unfavorable?",
        "What is the difference between a static budget variance and a flexible budget variance?",
    ],
)

_reg("accountancy", "Time Value of Money",
    explanation="the principle that a dollar today is worth more than a dollar in the future because it can be invested to earn interest or returns",
    reason="it underlies all financial decision-making including investment analysis, loan pricing, and retirement planning",
    concrete_examples=[
        "Receiving $1,000 today is better than $1,000 in 5 years because you can invest today's $1,000 and earn interest.",
        "A company comparing two projects must discount future cash flows to present value to make a fair comparison.",
        "Mortgage payments are calculated using TVM to spread a large sum into equal monthly payments over 30 years.",
    ],
    numerical_examples=[
        "Future Value: $1,000 at 5% for 3 years = $1,000 × (1.05)³ = $1,157.63.",
        "Present Value: $5,000 received in 4 years at 6% = $5,000 / (1.06)⁴ = $3,960.47.",
        "Annuity: Monthly payment on a $200,000 mortgage at 6% for 30 years = approximately $1,199/month.",
    ],
    misconceptions=[
        ("You can simply subtract inflation to convert future dollars to today's value",
         "Proper discounting uses compound interest mathematics, not simple subtraction. Subtracting inflation gives an approximation but becomes increasingly inaccurate for longer time periods and higher rates."),
    ],
    analogies=["Time value of money is like a seed — a dollar today is a seed you can plant, and it grows into more dollars over time through compounding."],
    prerequisites=["basic arithmetic", "percentages"],
    formulas=["FV = PV × (1 + r)ⁿ", "PV = FV / (1 + r)ⁿ"],
    document_evidence=[
        "The text explains: 'The time value of money is the most important concept in finance because it enables comparison of cash flows occurring at different times.'",
        "Table 14.1 provides present value factors for common interest rates and periods.",
    ],
    practice_problems=[
        "What is the present value of $10,000 to be received in 5 years if the discount rate is 8%?",
        "If you invest $5,000 today at 7% annual interest, how much will you have in 10 years?",
    ],
    hints=[
        "Use the FV formula to grow money forward in time, PV formula to bring it back to today.",
        "Higher discount rates make future money worth less today.",
    ],
    follow_up_questions=[
        "How does compounding frequency (annual vs. monthly) affect the future value?",
        "Why is the discount rate sometimes called the 'opportunity cost of capital'?",
    ],
    diagnostic_questions=[
        "Would you prefer $10,000 today or $12,000 in 3 years? What additional information do you need to decide?",
        "If interest rates rise, what happens to the present value of future cash flows?",
    ],
)

_reg("accountancy", "Inventory Valuation",
    explanation="the method used to assign costs to inventory items (FIFO, LIFO, or weighted average), which affects reported cost of goods sold and ending inventory value",
    reason="the chosen method impacts reported profits, taxes, and balance sheet values, especially when prices are changing",
    concrete_examples=[
        "Under FIFO (First-In, First-Out), the oldest inventory costs are expensed first. If early purchases were cheaper, FIFO reports higher profit.",
        "Under LIFO (Last-In, First-Out), the newest costs are expensed first. During rising prices, LIFO reports lower profit and lower taxes.",
        "Weighted average assigns the same average cost per unit to both COGS and ending inventory.",
    ],
    numerical_examples=[
        "Purchases: 100 units at $10, then 100 units at $14. Sold 120 units. FIFO COGS = (100×$10) + (20×$14) = $1,280. LIFO COGS = (100×$14) + (20×$10) = $1,600.",
        "Weighted average: Total cost = $2,400 / 200 units = $12/unit. COGS for 120 units = $1,440.",
        "The FIFO vs LIFO difference in COGS ($320) directly impacts pre-tax profit by the same amount.",
    ],
    misconceptions=[
        ("FIFO and LIFO refer to the physical flow of goods",
         "FIFO and LIFO are cost flow assumptions, not descriptions of how goods physically move through a warehouse. A company can ship the newest items first while using FIFO for costing purposes."),
    ],
    analogies=["Inventory valuation is like choosing which bills in your wallet to spend first — you can use the oldest ones (FIFO), the newest ones (LIFO), or treat them all as equally worn (weighted average)."],
    prerequisites=["basic arithmetic"],
    formulas=["COGS = Beginning Inventory + Purchases − Ending Inventory"],
    document_evidence=[
        "Section 7.4 states: 'Under rising prices, FIFO produces higher reported income and higher ending inventory values compared to LIFO.'",
        "Comparative Example 7.2 calculates COGS under all three methods using identical purchase data.",
    ],
    practice_problems=[
        "Given three purchases (50 units at $8, 60 units at $10, 40 units at $12) and 90 units sold, calculate COGS under FIFO and LIFO.",
        "Explain why a company might choose LIFO during a period of rising prices.",
    ],
    hints=[
        "FIFO: oldest costs go to COGS. LIFO: newest costs go to COGS.",
        "During rising prices: FIFO = higher profit, higher taxes; LIFO = lower profit, lower taxes.",
    ],
    follow_up_questions=[
        "Why is LIFO not permitted under International Financial Reporting Standards (IFRS)?",
        "How does the choice of inventory method affect the balance sheet?",
    ],
    diagnostic_questions=[
        "If prices are rising, which method reports higher net income: FIFO or LIFO?",
        "What does 'cost flow assumption' mean, and why does it matter?",
    ],
)

_reg("accountancy", "Tax Principles",
    explanation="the fundamental rules governing how taxes are assessed, including concepts like progressive taxation, deductions, credits, and the difference between tax avoidance and evasion",
    reason="understanding tax principles is essential for compliance, planning, and making informed financial decisions that minimize legal tax liability",
    concrete_examples=[
        "In a progressive tax system, someone earning $50,000 pays a lower effective rate than someone earning $200,000 because higher brackets are taxed at higher rates.",
        "A business deducting $10,000 in office rent reduces its taxable income by $10,000, not its tax bill by $10,000.",
        "A $2,000 tax credit reduces the actual tax owed by $2,000, dollar-for-dollar, unlike a deduction.",
    ],
    numerical_examples=[
        "With brackets of 10% up to $10,000 and 22% on $10,001-$40,000: tax on $30,000 income = ($10,000 × 0.10) + ($20,000 × 0.22) = $1,000 + $4,400 = $5,400.",
        "Effective tax rate = $5,400 / $30,000 = 18%, which is lower than the marginal rate of 22%.",
        "A $1,000 deduction for someone in the 22% bracket saves $220 in taxes, while a $1,000 credit saves the full $1,000.",
    ],
    misconceptions=[
        ("Moving into a higher tax bracket means ALL your income is taxed at the higher rate",
         "Tax brackets are marginal — only the income above the threshold is taxed at the higher rate. If the 22% bracket starts at $40,000, earning $41,000 means only the last $1,000 is taxed at 22%, not all $41,000."),
    ],
    analogies=["Progressive tax brackets are like filling stacked cups — the first cup fills at a low rate, the next at a higher rate, but only the water in each cup is taxed at that cup's rate."],
    prerequisites=["basic arithmetic", "percentages"],
    formulas=["Effective Tax Rate = Total Tax / Total Income", "Tax Savings from Deduction = Deduction Amount × Marginal Rate"],
    document_evidence=[
        "The text states: 'A tax deduction reduces taxable income, while a tax credit directly reduces the tax liability — credits are generally more valuable.'",
        "Table 15.1 shows current federal income tax brackets and rates.",
    ],
    practice_problems=[
        "Calculate the total tax and effective rate for someone earning $75,000 using the bracket: 10% up to $10,000, 12% from $10,001–$40,000, 22% from $40,001–$85,000.",
        "Compare the tax savings of a $5,000 deduction vs. a $5,000 credit for someone in the 24% bracket.",
    ],
    hints=[
        "Tax brackets are marginal — apply each rate only to the income within that bracket.",
        "Credits reduce tax owed; deductions reduce taxable income. Credits are almost always worth more.",
    ],
    follow_up_questions=[
        "What is the difference between tax avoidance (legal) and tax evasion (illegal)?",
        "Why do governments offer tax credits for certain activities like education or renewable energy?",
    ],
    diagnostic_questions=[
        "If someone says 'I don't want a raise because it'll put me in a higher tax bracket,' what misconception do they have?",
        "Is a $1,000 deduction or a $500 credit worth more to someone in the 30% bracket?",
    ],
)

_reg("accountancy", "Audit Procedures",
    explanation="the systematic methods auditors use to gather evidence and form an opinion on whether financial statements are fairly presented in accordance with accounting standards",
    reason="they provide assurance to stakeholders that financial information is reliable, complete, and free from material misstatement",
    concrete_examples=[
        "An auditor sends confirmation letters to a company's customers to verify that accounts receivable balances are accurate.",
        "Physical inventory counts involve auditors visiting warehouses to count and compare actual stock against recorded amounts.",
        "Analytical procedures compare current-year financial ratios to prior years and industry averages to identify unusual fluctuations.",
    ],
    numerical_examples=[
        "If the company reports $2 million in receivables but confirmations verify only $1.85 million, the $150,000 discrepancy requires investigation.",
        "An auditor sets materiality at 5% of net income. If net income is $500,000, any misstatement over $25,000 is material.",
        "Sampling: If the auditor tests 50 out of 5,000 invoices and finds 3 errors, the projected error rate is 6%, suggesting further investigation.",
    ],
    misconceptions=[
        ("An audit guarantees that the financial statements are 100% correct",
         "An audit provides 'reasonable assurance,' not absolute certainty. Auditors use sampling and professional judgment, and there is always some risk that a material misstatement goes undetected. Fraud deliberately designed to evade detection may escape even a well-conducted audit."),
    ],
    analogies=["An audit is like a health check-up — the doctor runs tests and examines samples but can't check every cell in your body. They provide reasonable confidence about your health, not a guarantee."],
    prerequisites=["basic arithmetic", "understanding of financial statements"],
    formulas=["Materiality Threshold = Percentage × Base Amount (e.g., 5% of net income)"],
    document_evidence=[
        "The auditing standards state: 'The auditor must obtain sufficient appropriate evidence to support the opinion on the financial statements.'",
        "ISA 500 outlines the types of audit evidence: inspection, observation, confirmation, recalculation, and inquiry.",
    ],
    practice_problems=[
        "If net income is $1,000,000 and materiality is set at 3%, what is the materiality threshold? What types of errors would be considered immaterial?",
        "An auditor finds that 4 out of 40 sampled transactions have documentation errors. What should the auditor do next?",
    ],
    hints=[
        "Audit evidence should be both sufficient (enough) and appropriate (relevant and reliable).",
        "The higher the risk, the more evidence the auditor needs to gather.",
    ],
    follow_up_questions=[
        "What is the difference between an internal audit and an external audit?",
        "What are the auditor's options if they discover a material misstatement that management refuses to correct?",
    ],
    diagnostic_questions=[
        "Why do auditors use sampling rather than checking every single transaction?",
        "What does 'reasonable assurance' mean in the context of an audit?",
    ],
)

# --- MATHEMATICS ---

_reg("mathematics", "Quadratic Equations",
    explanation="polynomial equations of the form ax² + bx + c = 0, where a ≠ 0, which have at most two solutions that can be found by factoring, completing the square, or using the quadratic formula",
    reason="they model parabolic trajectories, optimization problems, and appear throughout physics, engineering, and economics",
    concrete_examples=[
        "A ball thrown upward follows the path h(t) = −16t² + 64t + 5. Setting h = 0 gives a quadratic equation to find when the ball hits the ground.",
        "Finding two numbers that multiply to 12 and add to 7 means solving x² − 7x + 12 = 0.",
        "The area of a rectangle with perimeter 20 is maximized by solving a quadratic: A = x(10 − x).",
    ],
    numerical_examples=[
        "Solve 2x² − 5x − 3 = 0: Using the quadratic formula, x = (5 ± √(25+24))/4 = (5 ± 7)/4, giving x = 3 or x = −0.5.",
        "Factor x² − 9 = 0: (x − 3)(x + 3) = 0, so x = 3 or x = −3.",
        "Discriminant of 3x² + 2x + 5 = 0: b² − 4ac = 4 − 60 = −56 < 0, so there are no real solutions.",
    ],
    misconceptions=[
        ("Every quadratic equation has two different real solutions",
         "The discriminant (b² − 4ac) determines the number of solutions. If it's positive, there are two distinct real roots. If zero, there is exactly one repeated root. If negative, there are no real roots (only complex ones)."),
    ],
    analogies=["A quadratic equation is like a parabolic bridge — it touches the ground (x-axis) at zero, one, or two points depending on its height and shape."],
    prerequisites=["algebra", "square roots"],
    formulas=["x = (−b ± √(b²−4ac)) / 2a", "Discriminant = b² − 4ac"],
    document_evidence=[
        "The text states: 'The quadratic formula provides the solutions to any quadratic equation, whether or not it can be factored.'",
        "Example 2.5 demonstrates completing the square for x² + 6x + 2 = 0.",
    ],
    practice_problems=[
        "Solve 3x² + 7x − 6 = 0 using the quadratic formula.",
        "Determine whether x² + 4x + 5 = 0 has real solutions without solving it.",
    ],
    hints=[
        "Check the discriminant first to know how many solutions to expect.",
        "If a = 1 and b² − 4ac is a perfect square, try factoring first — it's faster.",
    ],
    follow_up_questions=[
        "How does the discriminant relate to the graph of the quadratic function?",
        "When would you choose completing the square over the quadratic formula?",
    ],
    diagnostic_questions=[
        "What are the three methods for solving quadratic equations?",
        "What does it mean geometrically when a quadratic has no real solutions?",
    ],
)

_reg("mathematics", "Derivatives",
    explanation="the instantaneous rate of change of a function, measuring how the output changes as the input changes by an infinitesimally small amount",
    reason="they enable optimization (finding maxima and minima), describe velocities and accelerations in physics, and underpin all of calculus",
    concrete_examples=[
        "The derivative of position with respect to time gives velocity: if s(t) = 5t², then v(t) = s'(t) = 10t.",
        "A company's marginal cost curve is the derivative of its total cost function.",
        "The slope of the tangent line to y = x³ at x = 2 is y' = 3(2²) = 12.",
    ],
    numerical_examples=[
        "If f(x) = 4x³ − 2x + 7, then f'(x) = 12x² − 2. At x = 1, f'(1) = 10.",
        "The derivative of sin(x) is cos(x). At x = π/3, the rate of change is cos(π/3) = 0.5.",
        "For f(x) = eˣ, f'(x) = eˣ. The function is its own derivative.",
    ],
    misconceptions=[
        ("If f'(x) = 0 then x is always a maximum",
         "f'(x) = 0 only means x is a critical point. It could be a maximum, minimum, or inflection point. You need the second derivative test or sign analysis to determine which. For example, f(x) = x³ has f'(0) = 0 but x = 0 is an inflection point."),
    ],
    analogies=["The derivative is like a speedometer — position tells you where you are, but the derivative (speedometer) tells you how fast you're changing position at this exact moment."],
    prerequisites=["algebra", "functions", "limits"],
    formulas=["Power rule: d/dx(xⁿ) = nxⁿ⁻¹", "Product rule: (fg)' = f'g + fg'", "Chain rule: d/dx[f(g(x))] = f'(g(x))·g'(x)"],
    document_evidence=[
        "The text defines: 'The derivative f'(a) equals the slope of the tangent line to the graph of f at the point (a, f(a)).'",
        "Theorem 3.1 states the power rule and provides a proof using limits.",
    ],
    practice_problems=[
        "Find the derivative of f(x) = 3x⁴ − 5x² + 2x − 1.",
        "Use the chain rule to differentiate g(x) = (2x + 1)⁵.",
    ],
    hints=[
        "For polynomials, bring the exponent down and subtract 1: d/dx(xⁿ) = nxⁿ⁻¹.",
        "The chain rule: differentiate the outside function, then multiply by the derivative of the inside.",
    ],
    follow_up_questions=[
        "What does the second derivative tell us about the shape of a function?",
        "How do derivatives help us find the maximum profit for a business?",
    ],
    diagnostic_questions=[
        "What is the geometric interpretation of the derivative at a point?",
        "If f'(x) > 0 on an interval, what does that tell you about f(x)?",
    ],
)

_reg("mathematics", "Integrals",
    explanation="the mathematical operation that computes the accumulated area under a curve, serving as the reverse of differentiation",
    reason="they enable calculation of total quantities from rates (distance from velocity, total revenue from marginal revenue) and are central to physics, engineering, and probability",
    concrete_examples=[
        "The integral of a velocity function gives the total distance traveled.",
        "The area under a demand curve between two prices gives consumer surplus.",
        "The integral of a probability density function over an interval gives the probability of that outcome.",
    ],
    numerical_examples=[
        "∫(2x)dx from 0 to 3 = [x²] from 0 to 3 = 9 − 0 = 9.",
        "∫(x² + 1)dx = x³/3 + x + C (indefinite integral).",
        "The area under y = 3x from x = 1 to x = 4 is ∫3x dx = [3x²/2] from 1 to 4 = 24 − 1.5 = 22.5.",
    ],
    misconceptions=[
        ("Integration always gives a positive area",
         "When the function is below the x-axis, the integral is negative. The integral of sin(x) from 0 to 2π is zero because the positive and negative areas cancel. To find total physical area, you must integrate the absolute value."),
    ],
    analogies=["Integration is like measuring how much water fills an oddly-shaped swimming pool — you add up infinitely thin slices of water (each with its own depth) to get the total volume."],
    prerequisites=["algebra", "derivatives"],
    formulas=["∫xⁿ dx = xⁿ⁺¹/(n+1) + C (n ≠ −1)", "Fundamental Theorem: ∫ₐᵇ f(x)dx = F(b) − F(a)"],
    document_evidence=[
        "The text states: 'The Fundamental Theorem of Calculus connects differentiation and integration, showing they are inverse operations.'",
        "Example 5.3 demonstrates the substitution method for integrating composite functions.",
    ],
    practice_problems=[
        "Evaluate ∫(3x² − 4x + 1)dx from 0 to 2.",
        "Find the indefinite integral of 5e²ˣ.",
    ],
    hints=[
        "The power rule for integration: add 1 to the exponent and divide by the new exponent.",
        "Always add the constant C for indefinite integrals.",
    ],
    follow_up_questions=[
        "What is the difference between definite and indefinite integrals?",
        "How does substitution simplify complex integrals?",
    ],
    diagnostic_questions=[
        "If F'(x) = f(x), what is the relationship between F and the integral of f?",
        "Can you always find an antiderivative in closed form? Give an example where you can't.",
    ],
)

_reg("mathematics", "Probability Distributions",
    explanation="mathematical functions that describe the likelihood of every possible outcome of a random experiment, specifying probabilities for discrete outcomes or probability densities for continuous outcomes",
    reason="they form the foundation of statistical inference, risk assessment, quality control, and machine learning",
    concrete_examples=[
        "A fair die has a uniform distribution: each face has probability 1/6.",
        "Heights of adult women follow approximately a normal distribution with mean ~64 inches and standard deviation ~3 inches.",
        "The number of emails received per hour follows a Poisson distribution if emails arrive independently at a constant average rate.",
    ],
    numerical_examples=[
        "For a binomial distribution with n=10, p=0.3: P(X=3) = C(10,3)(0.3)³(0.7)⁷ ≈ 0.267.",
        "For a normal distribution N(100, 15²): P(X > 130) corresponds to z = (130−100)/15 = 2, and P(Z > 2) ≈ 0.023.",
        "Poisson with λ=4: P(X=2) = e⁻⁴ × 4² / 2! ≈ 0.146.",
    ],
    misconceptions=[
        ("The probability of A and B is always P(A) × P(B)",
         "The multiplication rule P(A and B) = P(A) × P(B) only works when A and B are INDEPENDENT. For dependent events, P(A and B) = P(A) × P(B|A). Drawing cards without replacement is a classic example where independence fails."),
    ],
    analogies=["A probability distribution is like a blueprint for a weighted dart board — it tells you exactly how likely the dart is to land in each region."],
    prerequisites=["basic algebra", "combinatorics"],
    formulas=["P(X = k) for binomial: C(n,k)pᵏ(1−p)ⁿ⁻ᵏ", "Normal: z = (x − μ) / σ"],
    document_evidence=[
        "The text explains: 'The normal distribution, also called the bell curve, is characterized by its mean and standard deviation.'",
        "Table A.1 provides standard normal z-table values for finding probabilities.",
    ],
    practice_problems=[
        "A fair coin is flipped 8 times. What is the probability of getting exactly 5 heads?",
        "Scores on a test are normally distributed with mean 75 and SD 10. What proportion of students scored above 90?",
    ],
    hints=[
        "For the normal distribution, convert to z-scores first, then use the z-table.",
        "The binomial distribution requires: fixed n trials, constant p, independent trials, two outcomes.",
    ],
    follow_up_questions=[
        "When is it appropriate to approximate the binomial with a normal distribution?",
        "What does the Central Limit Theorem tell us about sample means?",
    ],
    diagnostic_questions=[
        "What is the difference between a discrete and continuous probability distribution?",
        "Why must the total area under a probability density function equal 1?",
    ],
)

_reg("mathematics", "Linear Algebra",
    explanation="the branch of mathematics dealing with vectors, vector spaces, matrices, and linear transformations, focusing on systems of linear equations",
    reason="it is essential for computer graphics, machine learning, quantum mechanics, engineering, and any field that manipulates multi-dimensional data",
    concrete_examples=[
        "Solving 3 equations with 3 unknowns (prices of 3 items given 3 budget constraints) is a system of linear equations solvable with matrices.",
        "Image rotation in computer graphics uses a 2×2 rotation matrix to transform every pixel's coordinates.",
        "Google's PageRank algorithm uses eigenvectors of a massive matrix to rank web pages.",
    ],
    numerical_examples=[
        "Matrix multiplication: [[1,2],[3,4]] × [[5],[6]] = [[1×5+2×6],[3×5+4×6]] = [[17],[39]].",
        "Determinant of [[3,1],[2,4]] = 3×4 − 1×2 = 10. Since det ≠ 0, the matrix is invertible.",
        "Eigenvalues of [[2,1],[0,3]]: det(A − λI) = (2−λ)(3−λ) = 0 → λ = 2 or λ = 3.",
    ],
    misconceptions=[
        ("Matrix multiplication is the same as multiplying corresponding elements",
         "Matrix multiplication uses dot products of rows and columns, not element-wise multiplication. If A is m×n and B is n×p, the (i,j) entry of AB is the dot product of row i of A and column j of B. Also, AB ≠ BA in general."),
    ],
    analogies=["A matrix is like a transformation machine — you feed in a vector (input), and the matrix stretches, rotates, or reflects it to produce a new vector (output)."],
    prerequisites=["algebra", "systems of equations"],
    formulas=["Ax = b (system of equations)", "det([[a,b],[c,d]]) = ad − bc"],
    document_evidence=[
        "The text states: 'A system of linear equations can be written in matrix form Ax = b, and solved when A is invertible.'",
        "Theorem 4.2: A square matrix is invertible if and only if its determinant is non-zero.",
    ],
    practice_problems=[
        "Multiply [[2,0],[1,3]] by [[1,4],[2,5]] and verify the result.",
        "Find the determinant of [[5,3],[2,7]] and determine whether the matrix is invertible.",
    ],
    hints=[
        "For 2×2 matrices: det = ad − bc. If det = 0, the matrix is singular (not invertible).",
        "Matrix multiplication is NOT commutative: AB ≠ BA in general.",
    ],
    follow_up_questions=[
        "What is an eigenvector, and why is it important in data science?",
        "How do you solve Ax = b when A is not invertible?",
    ],
    diagnostic_questions=[
        "What does it mean for a matrix to be 'invertible'?",
        "Can you multiply a 2×3 matrix by a 2×2 matrix? Why or why not?",
    ],
)

_reg("mathematics", "Trigonometric Identities",
    explanation="equations involving trigonometric functions that are true for all valid input values, used to simplify expressions and solve equations",
    reason="they simplify complex calculations in physics, engineering, signal processing, and any field involving periodic phenomena",
    concrete_examples=[
        "The identity sin²θ + cos²θ = 1 lets you find cosθ if you know sinθ without needing a calculator.",
        "In physics, expressing a wave as Asin(ωt + φ) and using sum identities lets you combine two waves into one.",
        "The double-angle formula sin(2θ) = 2sinθcosθ simplifies area calculations in trigonometry.",
    ],
    numerical_examples=[
        "If sinθ = 3/5, then cos²θ = 1 − 9/25 = 16/25, so cosθ = ±4/5.",
        "cos(π/3 + π/4) = cos(π/3)cos(π/4) − sin(π/3)sin(π/4) = (1/2)(√2/2) − (√3/2)(√2/2) = (√2 − √6)/4.",
        "tan(2×30°) = 2tan30°/(1 − tan²30°) = 2(1/√3)/(1 − 1/3) = (2/√3)/(2/3) = 3/√3 = √3.",
    ],
    misconceptions=[
        ("sin(A + B) = sinA + sinB",
         "The correct identity is sin(A + B) = sinAcosB + cosAsinB. Simply adding the sines gives incorrect results. For example, sin(30° + 60°) = sin(90°) = 1, but sin30° + sin60° = 0.5 + 0.866 = 1.366 ≠ 1."),
    ],
    analogies=["Trig identities are like algebraic shortcuts — knowing that sin²θ + cos²θ = 1 is like knowing that a² − b² = (a−b)(a+b), it lets you simplify without starting from scratch."],
    prerequisites=["algebra", "basic trigonometry"],
    formulas=["sin²θ + cos²θ = 1", "sin(A+B) = sinAcosB + cosAsinB", "cos(2θ) = cos²θ − sin²θ"],
    document_evidence=[
        "The text lists the fundamental identities: Pythagorean, sum/difference, double-angle, and half-angle.",
        "Proof 6.1 derives the sum formula sin(A+B) from the unit circle.",
    ],
    practice_problems=[
        "Simplify: (1 − cos²x) / sinx.",
        "Prove that tan²θ + 1 = sec²θ starting from sin²θ + cos²θ = 1.",
    ],
    hints=[
        "Start with sin²θ + cos²θ = 1 and divide both sides to get other identities.",
        "For sum/difference identities, don't try to add trig functions directly.",
    ],
    follow_up_questions=[
        "How are trig identities used in Fourier analysis?",
        "What is the practical use of the double-angle formulas?",
    ],
    diagnostic_questions=[
        "Is sin(2x) equal to 2sin(x)? Explain.",
        "What is the Pythagorean identity and where does it come from?",
    ],
)

_reg("mathematics", "Logarithms",
    explanation="the inverse of exponentiation — logₐ(x) answers the question 'to what power must a be raised to get x?'",
    reason="they transform multiplicative relationships into additive ones, making it easier to work with exponential growth, sound intensity, pH, and computational complexity",
    concrete_examples=[
        "log₁₀(1000) = 3 because 10³ = 1000.",
        "The Richter scale uses logarithms: an earthquake of magnitude 6 is 10 times stronger than magnitude 5.",
        "In computer science, binary search runs in O(log₂ n) time because it halves the search space each step.",
    ],
    numerical_examples=[
        "log₂(32) = 5 because 2⁵ = 32.",
        "Using the product rule: log₁₀(500) = log₁₀(5 × 100) = log₁₀(5) + log₁₀(100) = 0.699 + 2 = 2.699.",
        "Solving 2ˣ = 10: x = log₂(10) = ln(10)/ln(2) ≈ 3.322.",
    ],
    misconceptions=[
        ("log(A + B) = log(A) + log(B)",
         "The correct rule is log(A × B) = log(A) + log(B). Logarithms convert multiplication to addition, not addition to addition. For example, log(2 + 3) = log(5) ≈ 0.699, but log(2) + log(3) = 0.301 + 0.477 = 0.778 ≠ 0.699."),
    ],
    analogies=["A logarithm is like counting the number of digits in a number — log₁₀ essentially tells you 'how many digits?' minus 1. The number 1000 has 4 digits, and log₁₀(1000) = 3."],
    prerequisites=["algebra", "exponents"],
    formulas=["logₐ(xy) = logₐ(x) + logₐ(y)", "logₐ(xⁿ) = n·logₐ(x)", "Change of base: logₐ(x) = ln(x)/ln(a)"],
    document_evidence=[
        "The text states: 'Logarithms are the inverse of exponential functions: if aˣ = y, then logₐ(y) = x.'",
        "Table 7.1 summarizes the three key logarithm laws: product, quotient, and power rules.",
    ],
    practice_problems=[
        "Simplify log₂(16) + log₂(8) using logarithm rules.",
        "Solve for x: 3ˣ = 81.",
    ],
    hints=[
        "logₐ(x) asks 'a to what power equals x?'",
        "The product rule converts multiplication inside the log to addition outside.",
    ],
    follow_up_questions=[
        "Why is the natural logarithm (base e) preferred in calculus?",
        "How are logarithmic scales useful for data spanning many orders of magnitude?",
    ],
    diagnostic_questions=[
        "What is log₁₀(100) and how do you know?",
        "If log₃(x) = 4, what is x?",
    ],
)

_reg("mathematics", "Exponential Functions",
    explanation="functions of the form f(x) = aˣ where the variable is in the exponent, modeling quantities that grow or decay at a rate proportional to their current value",
    reason="they describe population growth, radioactive decay, compound interest, and the spread of diseases — any process where the rate of change is proportional to the current amount",
    concrete_examples=[
        "Bacteria doubling every hour follow exponential growth: starting with 100, after 5 hours you have 100 × 2⁵ = 3,200.",
        "Radioactive carbon-14 decays exponentially with a half-life of 5,730 years.",
        "Compound interest: $1,000 invested at 5% annual interest grows to $1,000(1.05)ⁿ after n years.",
    ],
    numerical_examples=[
        "A population of 500 growing at 3% per year: P(10) = 500 × (1.03)¹⁰ = 500 × 1.344 = 672.",
        "Half-life problem: If 200g of a substance has a half-life of 10 years, after 30 years: 200 × (1/2)³ = 25g remains.",
        "Continuous compounding: $5,000 at 4% for 6 years = $5,000 × e^(0.04×6) = $5,000 × e^0.24 ≈ $6,356.",
    ],
    misconceptions=[
        ("Exponential growth continues forever in real systems",
         "Exponential growth is always limited by resources, space, or other constraints. Real populations follow logistic growth (S-shaped), not pure exponential. Exponential models are accurate only in early stages."),
    ],
    analogies=["Exponential growth is like a chain letter — each person sends to 5 friends, who each send to 5 more. The numbers explode because growth feeds on itself."],
    prerequisites=["algebra", "exponents"],
    formulas=["f(x) = a·bˣ", "Continuous: f(t) = a·eᵏᵗ", "Half-life: t₁/₂ = ln(2)/k"],
    document_evidence=[
        "The text states: 'The number e ≈ 2.718 is the base of the natural exponential function, which arises naturally in continuous growth and decay.'",
        "Example 8.2 compares discrete and continuous compounding of investments.",
    ],
    practice_problems=[
        "A town of 10,000 grows at 2% per year. What is the population after 15 years?",
        "A substance decays according to N(t) = 100e^(−0.05t). How long until half remains?",
    ],
    hints=[
        "For growth, the base (or rate) must be greater than 1 (or positive). For decay, less than 1 (or negative).",
        "Half-life: set N(t) = N₀/2 and solve for t.",
    ],
    follow_up_questions=[
        "Why is e the 'natural' base for exponential functions?",
        "How do you convert between discrete (compound) and continuous growth models?",
    ],
    diagnostic_questions=[
        "If a quantity doubles every 5 years, how long until it's 8 times its original size?",
        "What is the difference between linear growth and exponential growth?",
    ],
)

_reg("mathematics", "Matrix Operations",
    explanation="the arithmetic operations that can be performed on matrices, including addition, scalar multiplication, matrix multiplication, transposition, and finding inverses and determinants",
    reason="they enable solving systems of equations, performing transformations in graphics and data science, and are the computational backbone of machine learning",
    concrete_examples=[
        "Adding two 2×2 matrices: [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]] — add corresponding elements.",
        "Scalar multiplication: 3 × [[1,2],[3,4]] = [[3,6],[9,12]].",
        "The transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]] — rows become columns.",
    ],
    numerical_examples=[
        "[[2,1],[0,3]] × [[1,4],[2,5]] = [[(2×1+1×2),(2×4+1×5)],[(0×1+3×2),(0×4+3×5)]] = [[4,13],[6,15]].",
        "Inverse of [[2,1],[5,3]]: det = 6−5 = 1. Inverse = [[3,−1],[−5,2]].",
        "Trace of [[4,1,0],[2,5,3],[0,1,6]] = 4 + 5 + 6 = 15.",
    ],
    misconceptions=[
        ("Matrix multiplication works like regular number multiplication — it's commutative",
         "Matrix multiplication is NOT commutative: AB ≠ BA in general. The order matters because row-column dot products depend on which matrix comes first. Also, matrices must have compatible dimensions for multiplication."),
    ],
    analogies=["Matrix operations are like choreographed group dances — adding matrices is everyone stepping in unison, but multiplication is an intricate routine where each row partners with each column."],
    prerequisites=["algebra", "arithmetic"],
    formulas=["(AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ", "A⁻¹ for 2×2: (1/det)[[d,−b],[−c,a]]"],
    document_evidence=[
        "The text states: 'For matrix multiplication AB to be defined, the number of columns of A must equal the number of rows of B.'",
        "Algorithm 4.1 provides the step-by-step procedure for Gaussian elimination.",
    ],
    practice_problems=[
        "Compute [[1,3],[2,4]] × [[2,0],[1,5]]. Then compute the product in reverse order and show AB ≠ BA.",
        "Find the inverse of [[4,7],[2,6]] and verify by computing AA⁻¹.",
    ],
    hints=[
        "For multiplication, remember: (m×n) × (n×p) = (m×p). Inner dimensions must match.",
        "To find the inverse of a 2×2 matrix, swap the diagonal, negate the off-diagonal, and divide by the determinant.",
    ],
    follow_up_questions=[
        "When is matrix multiplication commutative?",
        "What is the significance of the identity matrix?",
    ],
    diagnostic_questions=[
        "Can you multiply a 3×2 matrix by a 3×4 matrix? Why or why not?",
        "What condition must be met for a matrix to have an inverse?",
    ],
)

_reg("mathematics", "Statistical Inference",
    explanation="the process of drawing conclusions about a population based on data from a sample, using probability theory to quantify uncertainty",
    reason="it enables evidence-based decision-making in science, medicine, business, and public policy when studying entire populations is impractical",
    concrete_examples=[
        "A poll surveying 1,000 voters estimates the election outcome for millions of voters, with a margin of error.",
        "A clinical trial testing a drug on 500 patients infers whether the drug works for all patients with the condition.",
        "Quality control: testing 50 items from a production run of 10,000 to estimate the defect rate.",
    ],
    numerical_examples=[
        "A sample of n=100 has mean x̄=72 and SD s=10. The 95% CI for the population mean is 72 ± 1.96(10/√100) = [70.04, 73.96].",
        "Hypothesis test: H₀: μ = 50, H₁: μ ≠ 50. With x̄=53, s=12, n=36: z = (53−50)/(12/√36) = 1.5. p-value = 0.134 > 0.05, so do not reject H₀.",
        "If 42 out of 200 sampled items are defective, p̂ = 0.21. The 95% CI for the defect rate is 0.21 ± 1.96√(0.21×0.79/200) = [0.154, 0.266].",
    ],
    misconceptions=[
        ("A p-value of 0.03 means there is a 3% probability that the null hypothesis is true",
         "The p-value is the probability of observing data this extreme IF the null hypothesis were true, not the probability that the null is true. The null is either true or false — p-values measure evidence against it, not its probability."),
    ],
    analogies=["Statistical inference is like tasting a spoonful of soup to judge the whole pot — if you stir well (random sampling), a small taste tells you a lot about the full batch."],
    prerequisites=["basic algebra", "probability"],
    formulas=["CI: x̄ ± z*(σ/√n)", "Test statistic: z = (x̄ − μ₀)/(σ/√n)"],
    document_evidence=[
        "The text states: 'A 95% confidence interval means that if we repeated the sampling procedure many times, approximately 95% of the intervals would contain the true population parameter.'",
        "Table 9.1 provides critical z-values for common confidence levels.",
    ],
    practice_problems=[
        "A sample of 64 students has a mean GPA of 3.2 and SD of 0.4. Construct a 99% confidence interval for the population mean GPA.",
        "Test whether the average weight of packages exceeds 500g, given a sample mean of 508g, SD=15g, n=25 at α=0.05.",
    ],
    hints=[
        "Larger samples → narrower confidence intervals (more precision).",
        "The p-value is compared to α: if p < α, reject the null hypothesis.",
    ],
    follow_up_questions=[
        "What is the relationship between confidence level and interval width?",
        "What is a Type I error and a Type II error?",
    ],
    diagnostic_questions=[
        "What does 'statistically significant' actually mean?",
        "Why is random sampling important for inference?",
    ],
)

# --- SCIENCE ---

_reg("science", "Photosynthesis",
    explanation="the process by which green plants and some organisms convert light energy, water, and carbon dioxide into glucose and oxygen using chlorophyll",
    reason="it is the foundation of nearly all food chains and produces the oxygen that most life on Earth depends on for respiration",
    concrete_examples=[
        "A leaf appears green because chlorophyll absorbs red and blue light for photosynthesis and reflects green light.",
        "Aquatic plants release visible oxygen bubbles when placed under bright light — direct evidence of photosynthesis.",
        "Farmers use greenhouses to provide optimal light and CO₂ conditions to maximize photosynthesis and crop yield.",
    ],
    numerical_examples=[
        "The balanced equation: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂ shows that 6 molecules of CO₂ produce 1 molecule of glucose.",
        "A typical leaf intercepts about 1% of incident sunlight for photosynthesis, converting it to chemical energy.",
        "A tree with 200,000 leaves might produce about 100 kg of oxygen per year through photosynthesis.",
    ],
    misconceptions=[
        ("Plants get their food from the soil",
         "Plants get their carbon (the main structural element) from CO₂ in the air, not from soil. Water comes from roots, but the mass of a growing tree comes overwhelmingly from atmospheric carbon dioxide converted via photosynthesis. Soil provides minerals, but not 'food' in the caloric sense."),
    ],
    analogies=["Photosynthesis is like a solar-powered factory — sunlight is the electricity, CO₂ and water are the raw materials, and glucose is the finished product stored as chemical energy."],
    prerequisites=["basic chemistry", "cell biology"],
    formulas=["6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂"],
    document_evidence=[
        "The text explains: 'The light-dependent reactions occur in the thylakoid membranes, producing ATP and NADPH, while the Calvin cycle in the stroma uses these to fix CO₂ into glucose.'",
        "Diagram 4.3 illustrates the two stages of photosynthesis and their locations within the chloroplast.",
    ],
    practice_problems=[
        "What are the reactants and products of photosynthesis? Where in the cell does each stage occur?",
        "If a plant is placed in a sealed container with CO₂ but no light, will it photosynthesize? Explain.",
    ],
    hints=[
        "Think of photosynthesis in two stages: light reactions (need light) and Calvin cycle (don't directly need light).",
        "The oxygen released comes from splitting water molecules, not from CO₂.",
    ],
    follow_up_questions=[
        "What happens to the rate of photosynthesis as light intensity increases?",
        "How do C3, C4, and CAM plants handle photosynthesis differently?",
    ],
    diagnostic_questions=[
        "Where does the carbon in a tree trunk originally come from?",
        "What would happen to Earth's atmosphere if photosynthesis stopped?",
    ],
)

_reg("science", "Newton's Laws",
    explanation="three fundamental laws describing the relationship between forces acting on objects and their resulting motion",
    reason="they form the foundation of classical mechanics and allow us to predict the motion of everything from falling apples to orbiting planets",
    concrete_examples=[
        "First Law: A hockey puck on ice slides a long way because there's little friction — an object in motion stays in motion.",
        "Second Law: Pushing a shopping cart harder (more force) makes it accelerate faster; loading it with groceries (more mass) makes it harder to push.",
        "Third Law: When you jump, your feet push down on the ground, and the ground pushes you upward with equal force.",
    ],
    numerical_examples=[
        "F = ma: A 5 kg box accelerated at 3 m/s² requires F = 5 × 3 = 15 N of force.",
        "Weight: A 70 kg astronaut on Earth: W = 70 × 9.8 = 686 N. On the Moon (g = 1.6): W = 70 × 1.6 = 112 N.",
        "Third Law: A 1000 kg car pushes the road backward with 3000 N. The road pushes the car forward with 3000 N.",
    ],
    misconceptions=[
        ("Objects in motion always experience a force in the direction of motion",
         "Newton's First Law says an object in motion stays in motion at constant velocity unless acted upon by an external force. A thrown ball continues forward not because of a forward force, but because nothing has stopped it yet. Gravity and air resistance are forces that change its motion."),
    ],
    analogies=["Newton's Laws are like rules of a board game — the first law says pieces stay put unless you move them, the second law says heavier pieces are harder to move, and the third law says every push creates an equal push back."],
    prerequisites=["basic algebra", "measurement units"],
    formulas=["F = ma", "W = mg", "Action = −Reaction"],
    document_evidence=[
        "The text states: 'Newton's Second Law, F = ma, is the quantitative relationship between net force, mass, and acceleration.'",
        "Experiment 3.1 demonstrates the First Law using an air track to minimize friction.",
    ],
    practice_problems=[
        "A 2000 kg car accelerates from 0 to 20 m/s in 10 seconds. What is the net force on the car?",
        "If you push a wall with 50 N of force, what force does the wall exert on you?",
    ],
    hints=[
        "F = ma requires NET force — add up all forces (including friction) before using the formula.",
        "Third Law pairs act on DIFFERENT objects — the action force on one and the reaction on the other.",
    ],
    follow_up_questions=[
        "If an object moves at constant velocity, what is the net force on it?",
        "How do Newton's Laws apply to objects in orbit?",
    ],
    diagnostic_questions=[
        "If no force acts on a moving object, does it slow down, speed up, or keep its speed?",
        "When a horse pulls a cart, the cart pulls the horse backward with equal force. Why does the cart still move forward?",
    ],
)

_reg("science", "Atomic Structure",
    explanation="the arrangement of subatomic particles (protons, neutrons, and electrons) within an atom, with protons and neutrons forming a dense nucleus orbited by electrons in energy levels",
    reason="it explains chemical bonding, the periodic table's organization, radioactivity, and all chemical reactions",
    concrete_examples=[
        "A carbon atom has 6 protons, 6 neutrons, and 6 electrons. The protons define it as carbon.",
        "Neon's full outer electron shell (8 electrons) makes it chemically inert — it doesn't react easily.",
        "Isotopes like Carbon-12 and Carbon-14 have the same proton count but different neutron counts.",
    ],
    numerical_examples=[
        "Oxygen-16: 8 protons + 8 neutrons = mass number 16. Atomic number = 8 (number of protons).",
        "If an atom has 11 electrons in a neutral state, it has 11 protons and is sodium (Na).",
        "Electron configuration of calcium (Z=20): 1s² 2s² 2p⁶ 3s² 3p⁶ 4s². Two electrons in the outermost shell.",
    ],
    misconceptions=[
        ("Electrons orbit the nucleus like planets orbit the sun",
         "The planetary model is an oversimplification. Electrons exist in probability clouds (orbitals) — regions where they are likely to be found. They don't follow fixed circular paths. The quantum mechanical model describes them as wave-like probability distributions."),
    ],
    analogies=["An atom is like a stadium — the nucleus is a marble at the center of the field, and the electrons are bees buzzing around in the upper decks, never in fixed seats."],
    prerequisites=["basic chemistry"],
    formulas=["Mass number = Protons + Neutrons", "Atomic number = Number of protons"],
    document_evidence=[
        "The text explains: 'The atomic number determines an element's identity, while isotopes of the same element differ only in neutron count.'",
        "Rutherford's gold foil experiment (Section 2.3) proved that atoms have a small, dense, positively charged nucleus.",
    ],
    practice_problems=[
        "An atom has 17 protons, 18 neutrons, and 17 electrons. Identify the element and its mass number.",
        "Write the electron configuration for phosphorus (Z=15) and determine how many valence electrons it has.",
    ],
    hints=[
        "Atomic number = protons = electrons (in neutral atom). It identifies the element.",
        "Electrons fill orbitals in order: 1s, 2s, 2p, 3s, 3p, 4s, 3d, ...",
    ],
    follow_up_questions=[
        "Why do elements in the same column of the periodic table have similar chemical properties?",
        "What is the difference between an ion and an isotope?",
    ],
    diagnostic_questions=[
        "If you change the number of protons in an atom, what happens?",
        "What determines how an atom bonds with other atoms?",
    ],
)

_reg("science", "Chemical Bonding",
    explanation="the attractive forces that hold atoms together in molecules or crystals, including ionic bonds (electron transfer), covalent bonds (electron sharing), and metallic bonds",
    reason="it determines the physical and chemical properties of substances — melting point, conductivity, reactivity, and shape",
    concrete_examples=[
        "Table salt (NaCl) forms ionic bonds: sodium donates an electron to chlorine, creating oppositely charged ions that attract.",
        "Water (H₂O) has covalent bonds: oxygen shares electrons with two hydrogen atoms.",
        "Metals like copper have metallic bonds where a 'sea' of electrons flows freely, enabling electrical conductivity.",
    ],
    numerical_examples=[
        "In NaCl: Na (11 electrons) loses 1 → Na⁺ (10 electrons). Cl (17 electrons) gains 1 → Cl⁻ (18 electrons).",
        "A carbon atom with 4 valence electrons forms 4 covalent bonds (e.g., methane CH₄), sharing one electron with each hydrogen.",
        "Electronegativity difference > 1.7 typically indicates an ionic bond; < 0.5 indicates nonpolar covalent.",
    ],
    misconceptions=[
        ("Ionic bonds are stronger than covalent bonds",
         "Bond strength depends on context. Individual covalent bonds (like C-C at 346 kJ/mol) can be very strong. Ionic crystals have high lattice energies (NaCl: 787 kJ/mol), but this is a collective property. In water, ionic compounds dissolve easily while covalent diamonds don't, showing strength depends on conditions."),
    ],
    analogies=["Ionic bonding is like giving a friend your jacket (transferring), while covalent bonding is like two friends sharing a blanket (sharing electrons)."],
    prerequisites=["atomic structure", "periodic table"],
    formulas=["Octet rule: atoms tend to gain/lose/share electrons to have 8 in their outer shell"],
    document_evidence=[
        "The text states: 'The type of bond formed depends primarily on the electronegativity difference between the bonding atoms.'",
        "Lewis dot structures (Section 5.2) visually represent how valence electrons are shared or transferred.",
    ],
    practice_problems=[
        "Draw the Lewis structure for CO₂ and identify the type of bonds present.",
        "Predict whether MgO is ionic or covalent, and explain your reasoning using electronegativity.",
    ],
    hints=[
        "Large electronegativity difference → ionic. Small → covalent.",
        "Use the octet rule as a guide, but know exceptions (H needs 2, expanded octets in period 3+).",
    ],
    follow_up_questions=[
        "Why does water dissolve salt but not oil?",
        "What is a polar covalent bond and how does it differ from nonpolar?",
    ],
    diagnostic_questions=[
        "What determines whether atoms will share or transfer electrons?",
        "Why do metals conduct electricity but ionic solids (when solid) do not?",
    ],
)

_reg("science", "DNA Replication",
    explanation="the biological process by which a cell copies its DNA before cell division, producing two identical DNA molecules from one original",
    reason="it ensures genetic information is faithfully passed from parent to daughter cells, enabling growth, repair, and reproduction",
    concrete_examples=[
        "Before a skin cell divides, it replicates all 3 billion base pairs of DNA so each new cell gets a complete copy.",
        "Errors in DNA replication can cause mutations, some of which may lead to cancer if they occur in genes controlling cell growth.",
        "PCR (polymerase chain reaction) mimics DNA replication in the lab, amplifying tiny DNA samples for forensic analysis.",
    ],
    numerical_examples=[
        "Human DNA replication proceeds at about 1,000 nucleotides per second per replication fork, completing 3 billion base pairs in roughly 8 hours using multiple origins.",
        "Replication error rate: DNA polymerase makes about 1 error per 10⁹ base pairs after proofreading and repair.",
        "Starting from one DNA molecule, after 30 rounds of replication (as in PCR), you get 2³⁰ ≈ 1 billion copies.",
    ],
    misconceptions=[
        ("DNA replication produces two entirely new DNA molecules",
         "Replication is semi-conservative: each new DNA molecule contains one original (parent) strand and one newly synthesized (daughter) strand. Meselson and Stahl's experiment confirmed this in 1958."),
    ],
    analogies=["DNA replication is like unzipping a zipper and building a new matching half for each side — each resulting zipper has one old rail and one new rail."],
    prerequisites=["basic cell biology", "DNA structure"],
    formulas=["Base pairing rules: A-T, G-C"],
    document_evidence=[
        "The text explains: 'Helicase unwinds the double helix, and DNA polymerase synthesizes new strands by adding complementary nucleotides.'",
        "The Meselson-Stahl experiment (Section 8.3) provided definitive evidence for semi-conservative replication.",
    ],
    practice_problems=[
        "Given a template strand 3'-ATGCCTA-5', write the sequence of the new complementary strand.",
        "Explain the roles of helicase, primase, and DNA polymerase in replication.",
    ],
    hints=[
        "Remember: A pairs with T, G pairs with C. The new strand is antiparallel to the template.",
        "Replication starts at origins and proceeds in both directions using replication forks.",
    ],
    follow_up_questions=[
        "What is the leading strand vs. the lagging strand, and why are they different?",
        "How do cells correct errors made during DNA replication?",
    ],
    diagnostic_questions=[
        "If one strand reads 5'-AATGCG-3', what does the complementary strand read?",
        "Why is DNA replication called 'semi-conservative'?",
    ],
)

_reg("science", "Cell Mitosis",
    explanation="the type of cell division that produces two genetically identical daughter cells from a single parent cell, used for growth, repair, and asexual reproduction",
    reason="it enables organisms to grow from a single fertilized egg, replace damaged tissues, and maintain body systems",
    concrete_examples=[
        "A scraped knee heals because skin cells undergo mitosis to replace lost cells.",
        "A tadpole growing into a frog involves massive amounts of mitosis as new tissues form.",
        "Cancer occurs when mitosis becomes uncontrolled, producing cells that divide without stopping.",
    ],
    numerical_examples=[
        "Starting with 1 cell, after 10 rounds of mitosis: 2¹⁰ = 1,024 cells.",
        "Human red blood cells are replaced at about 200 billion per day through mitosis of precursor cells.",
        "The cell cycle in human cells typically lasts 24 hours, with mitosis itself taking only about 1–2 hours.",
    ],
    misconceptions=[
        ("Mitosis and meiosis produce the same type of cells",
         "Mitosis produces two diploid (2n) cells identical to the parent, used for growth and repair. Meiosis produces four haploid (n) cells (gametes) with half the chromosomes and genetic variation through crossing over. They serve completely different purposes."),
    ],
    analogies=["Mitosis is like making a photocopy of a document — you start with one original and end up with two identical copies, one for each new cell."],
    prerequisites=["cell biology", "DNA structure"],
    formulas=["After n divisions: 2ⁿ cells"],
    document_evidence=[
        "The text describes the four phases: 'In prophase, chromosomes condense; in metaphase, they align at the cell's equator; in anaphase, sister chromatids separate; in telophase, nuclear envelopes reform.'",
        "Micrograph 6.4 shows cells at each stage of mitosis in an onion root tip.",
    ],
    practice_problems=[
        "List the four phases of mitosis in order and describe what happens in each.",
        "If a cell with 46 chromosomes undergoes mitosis, how many chromosomes does each daughter cell have?",
    ],
    hints=[
        "Remember PMAT: Prophase, Metaphase, Anaphase, Telophase.",
        "Mitosis = identical copies. Meiosis = variation and half the chromosomes.",
    ],
    follow_up_questions=[
        "What would happen if the spindle fibers failed during anaphase?",
        "How is mitosis regulated, and what goes wrong in cancer?",
    ],
    diagnostic_questions=[
        "At what phase do chromosomes line up at the middle of the cell?",
        "What is the difference between mitosis and cytokinesis?",
    ],
)

_reg("science", "Thermodynamics",
    explanation="the branch of physics dealing with heat, work, energy, and the laws governing energy transformation and transfer",
    reason="it explains why engines work, why heat flows from hot to cold, and why perpetual motion machines are impossible",
    concrete_examples=[
        "A car engine converts chemical energy (fuel) into mechanical energy (motion) and waste heat — illustrating the first and second laws.",
        "An ice cube melting in warm water: heat transfers from the water to the ice until thermal equilibrium.",
        "A refrigerator moves heat from cold (inside) to hot (outside) by doing work, consistent with the second law.",
    ],
    numerical_examples=[
        "First Law: If a gas absorbs 500 J of heat and does 200 J of work, its internal energy increases by 300 J (ΔU = Q − W).",
        "Carnot efficiency: A heat engine operating between 600 K and 300 K has maximum efficiency = 1 − 300/600 = 50%.",
        "Specific heat: Heating 2 kg of water from 20°C to 100°C requires Q = mcΔT = 2 × 4186 × 80 = 669,760 J ≈ 670 kJ.",
    ],
    misconceptions=[
        ("Cold is transferred from cold objects to warm objects",
         "There is no such thing as 'cold' being transferred. Heat (thermal energy) always flows from higher temperature to lower temperature. A cold object feels cold because heat is flowing OUT of your hand INTO the object, not because 'cold' is entering your hand."),
    ],
    analogies=["Thermodynamics is like banking — the first law says energy is conserved (you can't create money from nothing), and the second law says every transaction has a fee (entropy always increases)."],
    prerequisites=["basic algebra", "temperature scales"],
    formulas=["ΔU = Q − W (First Law)", "Efficiency = 1 − T_cold/T_hot (Carnot)", "Q = mcΔT"],
    document_evidence=[
        "The text states: 'The First Law of Thermodynamics is conservation of energy: energy cannot be created or destroyed, only converted.'",
        "The Second Law (Section 12.4): 'Heat spontaneously flows from hot to cold; the reverse requires work input.'",
    ],
    practice_problems=[
        "A system absorbs 800 J of heat and performs 350 J of work. What is the change in internal energy?",
        "Calculate the maximum efficiency of a steam engine operating between 400°C (673 K) and 25°C (298 K).",
    ],
    hints=[
        "First Law: energy in = energy out + change in internal energy.",
        "Entropy always increases in an isolated system — nature tends toward disorder.",
    ],
    follow_up_questions=[
        "Why can no real heat engine be 100% efficient?",
        "How does the concept of entropy relate to everyday experiences like ice melting?",
    ],
    diagnostic_questions=[
        "If you hold an ice cube, is 'cold' flowing into your hand? Explain what's actually happening.",
        "What does the Second Law of Thermodynamics tell us about the direction of natural processes?",
    ],
)

_reg("science", "Electromagnetic Spectrum",
    explanation="the complete range of electromagnetic radiation, from low-energy radio waves to high-energy gamma rays, all traveling at the speed of light but differing in wavelength and frequency",
    reason="it encompasses everything from radio communication to medical X-rays to visible light, making it fundamental to technology and understanding nature",
    concrete_examples=[
        "Your smartphone receives radio waves (long wavelength), your microwave oven uses microwaves, and your eyes detect visible light — all are electromagnetic radiation.",
        "UV radiation from the sun causes sunburn because its high energy damages skin cell DNA.",
        "X-rays pass through soft tissue but are absorbed by bone, creating medical images.",
    ],
    numerical_examples=[
        "Visible light spans wavelengths of approximately 400 nm (violet) to 700 nm (red).",
        "Speed of light: c = λf. For red light (λ = 700 nm): f = 3×10⁸ / 7×10⁻⁷ = 4.3×10¹⁴ Hz.",
        "Photon energy: E = hf. For UV light at f = 1×10¹⁵ Hz: E = 6.63×10⁻³⁴ × 10¹⁵ = 6.63×10⁻¹⁹ J.",
    ],
    misconceptions=[
        ("Radio waves and gamma rays are completely different types of energy",
         "They are all the SAME type of energy — electromagnetic radiation. The only difference is wavelength and frequency. Radio waves have long wavelengths and low energy; gamma rays have short wavelengths and high energy. They all travel at the speed of light in a vacuum."),
    ],
    analogies=["The EM spectrum is like a piano keyboard — radio waves are the deep bass notes (low frequency, long wavelength) and gamma rays are the high-pitched notes (high frequency, short wavelength), but they're all sound waves (well, light waves)."],
    prerequisites=["basic physics", "wave properties"],
    formulas=["c = λf", "E = hf"],
    document_evidence=[
        "The text organizes the spectrum: 'From longest to shortest wavelength: radio, microwave, infrared, visible, ultraviolet, X-ray, gamma ray.'",
        "Figure 7.2 displays the EM spectrum with wavelength ranges and common applications for each band.",
    ],
    practice_problems=[
        "Calculate the frequency of green light with wavelength 550 nm.",
        "Which has more energy per photon: infrared light or ultraviolet light? Explain using E = hf.",
    ],
    hints=[
        "Higher frequency = shorter wavelength = more energy per photon.",
        "Use c = λf to convert between wavelength and frequency (c = 3×10⁸ m/s).",
    ],
    follow_up_questions=[
        "Why do different materials absorb different parts of the EM spectrum?",
        "How do astronomers use the EM spectrum to learn about distant stars?",
    ],
    diagnostic_questions=[
        "Are radio waves and visible light fundamentally different kinds of energy?",
        "Why is UV radiation more dangerous than visible light?",
    ],
)

_reg("science", "Plate Tectonics",
    explanation="the scientific theory that Earth's lithosphere is divided into large plates that float on the semi-fluid asthenosphere and move slowly, causing earthquakes, volcanic activity, and mountain formation",
    reason="it explains the distribution of earthquakes and volcanoes, continental shapes, ocean floor features, and the long-term evolution of Earth's surface",
    concrete_examples=[
        "South America and Africa's coastlines fit together like puzzle pieces because they were once joined as part of Pangaea.",
        "The Himalayas formed from the collision of the Indian plate and the Eurasian plate — a convergent boundary.",
        "Iceland sits on the Mid-Atlantic Ridge, where the North American and Eurasian plates are pulling apart.",
    ],
    numerical_examples=[
        "The Atlantic Ocean widens by about 2.5 cm per year as the plates diverge — about the speed your fingernails grow.",
        "The Pacific Plate moves northwest at about 7–10 cm per year.",
        "The oldest oceanic crust is about 200 million years old, compared to continental crust up to 4 billion years old.",
    ],
    misconceptions=[
        ("Earthquakes happen because the ground suddenly opens up into a giant crack",
         "Earthquakes occur when stress built up along fault lines is suddenly released as the plates slip. The ground shakes from seismic waves, but it doesn't open into chasms. Buildings collapse from shaking, not from the ground splitting open."),
    ],
    analogies=["Tectonic plates are like giant crackers floating on thick soup — they drift, bump into each other (causing earthquakes), spread apart (creating new crust), and sometimes slide under one another."],
    prerequisites=["basic geology", "Earth's layers"],
    formulas=["Plate velocity ≈ distance / time (geological timescale)"],
    document_evidence=[
        "The text states: 'Evidence for plate tectonics includes matching fossils on separated continents, seafloor spreading magnetic stripes, and GPS measurements of plate motion.'",
        "Map 10.1 shows the major tectonic plates and the types of boundaries between them.",
    ],
    practice_problems=[
        "If the Atlantic Ocean is 5,000 km wide and opened at 2.5 cm/year, approximately how long ago did it start forming?",
        "Classify the following as convergent, divergent, or transform: Himalayas, Mid-Atlantic Ridge, San Andreas Fault.",
    ],
    hints=[
        "Three boundary types: convergent (collide), divergent (separate), transform (slide past).",
        "Fossils of the same species on different continents = evidence they were once connected.",
    ],
    follow_up_questions=[
        "What drives the movement of tectonic plates?",
        "Why is oceanic crust younger than continental crust?",
    ],
    diagnostic_questions=[
        "What evidence convinced scientists that continents move?",
        "What type of plate boundary creates new oceanic crust?",
    ],
)

_reg("science", "Natural Selection",
    explanation="the process by which organisms with traits better suited to their environment tend to survive and reproduce more, passing those advantageous traits to the next generation",
    reason="it is the primary mechanism of evolution, explaining how species adapt to their environments and how biodiversity arises",
    concrete_examples=[
        "Peppered moths in industrial England: dark-colored moths survived better on soot-darkened trees, shifting the population from mostly light to mostly dark.",
        "Antibiotic-resistant bacteria: when antibiotics kill susceptible bacteria, resistant ones survive and multiply.",
        "Darwin's finches: different beak shapes evolved on different Galápagos islands to exploit different food sources.",
    ],
    numerical_examples=[
        "If 10% of a beetle population has a camouflage mutation and predators eat 80% of non-camouflaged beetles but only 20% of camouflaged ones, the camouflaged fraction grows rapidly across generations.",
        "If a trait provides a 1% survival advantage, it can spread through a population of 10,000 in roughly 1,000 generations.",
        "Hardy-Weinberg: if p² + 2pq + q² = 1 and q = 0.3, then q² = 0.09 (9% show the recessive phenotype).",
    ],
    misconceptions=[
        ("Organisms evolve on purpose to adapt to their environment",
         "Evolution has no purpose or direction. Random mutations occur, and natural selection acts on existing variation. Organisms don't 'choose' to evolve. Those with beneficial mutations survive more — it's a filtering process, not a goal-directed one."),
    ],
    analogies=["Natural selection is like a sieve — all varieties of organisms pass through, but only those whose traits fit through the environmental 'holes' survive and reproduce."],
    prerequisites=["basic biology", "genetics"],
    formulas=["Fitness = reproductive success relative to other individuals in the population"],
    document_evidence=[
        "The text states: 'Natural selection requires: variation in traits, heritability of those traits, and differential fitness based on traits.'",
        "Darwin's observations in the Galápagos (Section 11.2) provided key evidence for natural selection.",
    ],
    practice_problems=[
        "Explain how antibiotic resistance in bacteria is an example of natural selection. Identify the variation, selection pressure, and outcome.",
        "Why doesn't natural selection produce 'perfect' organisms?",
    ],
    hints=[
        "Three requirements: variation exists, traits are heritable, and some variants survive/reproduce better.",
        "Natural selection acts on existing variation — it cannot create new traits from scratch.",
    ],
    follow_up_questions=[
        "What is the difference between natural selection and genetic drift?",
        "Can natural selection lead to the loss of traits? Give an example.",
    ],
    diagnostic_questions=[
        "Did giraffes evolve long necks because they 'needed' to reach high leaves? Explain the correct mechanism.",
        "What are the three conditions necessary for natural selection to occur?",
    ],
)

# --- NUTRITION / FOOD SCIENCE ---

_reg("nutrition_food_science", "Macronutrients",
    explanation="the three main categories of nutrients — carbohydrates, proteins, and fats — that provide energy (calories) and serve structural and functional roles in the body",
    reason="understanding macronutrients is essential for balanced nutrition, disease prevention, and athletic performance",
    concrete_examples=[
        "Carbohydrates in rice and bread provide quick energy (4 cal/g) for brain function and exercise.",
        "Proteins in chicken and beans build and repair muscles and tissues (4 cal/g).",
        "Fats in avocados and olive oil store energy (9 cal/g), insulate organs, and absorb fat-soluble vitamins.",
    ],
    numerical_examples=[
        "A meal with 60g carbs, 30g protein, and 20g fat provides: (60×4) + (30×4) + (20×9) = 240 + 120 + 180 = 540 calories.",
        "An athlete needing 3,000 cal/day at 50/30/20 split: 375g carbs, 225g protein, 67g fat.",
        "A tablespoon of olive oil (14g fat) provides 14 × 9 = 126 calories.",
    ],
    misconceptions=[
        ("All fats are unhealthy and should be avoided",
         "Fats are essential for absorbing vitamins A, D, E, K, for brain function, and for hormone production. Unsaturated fats (found in fish, nuts, olive oil) are heart-healthy. Only trans fats and excessive saturated fats are associated with health risks."),
    ],
    analogies=["Macronutrients are like the three types of fuel for a car — carbs are gasoline (quick burn), fats are diesel (slow, sustained burn), and proteins are repair kits (fix and build parts)."],
    prerequisites=["basic biology", "chemistry basics"],
    formulas=["Carbs: 4 cal/g", "Protein: 4 cal/g", "Fat: 9 cal/g", "Total calories = sum of all macronutrient calories"],
    document_evidence=[
        "The text states: 'Each macronutrient plays a distinct role: carbohydrates provide primary energy, proteins build structure, and fats store energy and support cell membranes.'",
        "Table 1.1 shows recommended daily macronutrient ranges from dietary guidelines.",
    ],
    practice_problems=[
        "A nutrition label shows 45g carbs, 12g protein, and 8g fat per serving. Calculate the total calories.",
        "If someone needs 2,500 calories/day with 55% from carbs, how many grams of carbs should they eat?",
    ],
    hints=[
        "Fats have more than double the calories per gram compared to carbs or protein.",
        "To convert percentage of calories to grams: (total cal × %) / cal per gram of that macro.",
    ],
    follow_up_questions=[
        "Why do athletes often 'carb-load' before endurance events?",
        "What happens if you consume too much protein — does it all become muscle?",
    ],
    diagnostic_questions=[
        "Which macronutrient provides the most calories per gram?",
        "Are all types of fat equally bad for health? Explain.",
    ],
)

_reg("nutrition_food_science", "Glycemic Index",
    explanation="a numerical scale (0–100) that ranks carbohydrate-containing foods by how quickly they raise blood sugar levels after eating, with glucose set at 100",
    reason="it helps people with diabetes manage blood sugar, and guides food choices for sustained energy and weight management",
    concrete_examples=[
        "White bread has a high GI (~75), causing rapid blood sugar spikes, while whole-grain bread has a lower GI (~50).",
        "An apple (GI ~36) releases sugar slowly, providing sustained energy, unlike apple juice (GI ~41) which is faster.",
        "Athletes may choose high-GI foods after exercise for quick recovery, but low-GI foods before exercise for sustained energy.",
    ],
    numerical_examples=[
        "GI of white rice: ~73 (high). GI of brown rice: ~50 (low-medium). A diabetic choosing brown rice would experience a slower, lower blood sugar peak.",
        "Glycemic load = (GI × grams of carbs per serving) / 100. For a slice of white bread (GI 75, 15g carbs): GL = 75×15/100 = 11.25.",
        "A meal with GL < 10 is low glycemic load, 10-20 is medium, >20 is high.",
    ],
    misconceptions=[
        ("All sugar causes the same blood sugar spike",
         "Different forms of sugar and different foods containing sugar raise blood sugar at very different rates. Fructose (fruit sugar) has a much lower GI (~20) than glucose (100). The food matrix — fiber, fat, protein content — significantly slows sugar absorption."),
    ],
    analogies=["The glycemic index is like a speed limit for blood sugar — high-GI foods rush sugar into the bloodstream like a highway, while low-GI foods release it slowly like a winding country road."],
    prerequisites=["basic biology", "carbohydrate chemistry"],
    formulas=["Glycemic Load = (GI × carbs per serving in grams) / 100"],
    document_evidence=[
        "The text explains: 'The glycemic index measures how quickly a food raises blood glucose compared to pure glucose.'",
        "Table 3.2 lists GI values for common foods including white bread (75), apple (36), and lentils (32).",
    ],
    practice_problems=[
        "Compare the glycemic load of a banana (GI 51, 27g carbs) versus a chocolate bar (GI 44, 30g carbs). Which has a higher glycemic load?",
        "Why might a low-GI diet be beneficial for someone with Type 2 diabetes?",
    ],
    hints=[
        "GI measures how FAST blood sugar rises. Glycemic LOAD also considers HOW MUCH carb you eat.",
        "Fiber, fat, and protein in a meal slow down sugar absorption, lowering the effective GI.",
    ],
    follow_up_questions=[
        "How does cooking method affect a food's glycemic index?",
        "What is the difference between glycemic index and glycemic load, and why does it matter?",
    ],
    diagnostic_questions=[
        "If a food has a high GI but you eat only a tiny portion, will your blood sugar spike? Explain.",
        "Why do whole grains tend to have a lower GI than refined grains?",
    ],
)

_reg("nutrition_food_science", "Protein Synthesis",
    explanation="the cellular process by which the information in DNA is used to build proteins through transcription (DNA → mRNA) and translation (mRNA → protein at ribosomes)",
    reason="proteins carry out nearly all functions in the body — from enzymes to hormones to antibodies — making protein synthesis fundamental to life",
    concrete_examples=[
        "Your body synthesizes hemoglobin protein in bone marrow cells to carry oxygen in red blood cells.",
        "After exercise, muscle cells ramp up protein synthesis to repair and grow muscle fibers using amino acids from dietary protein.",
        "Insulin is a protein hormone synthesized by pancreatic beta cells from the INS gene.",
    ],
    numerical_examples=[
        "A ribosome translates mRNA at about 6 amino acids per second in eukaryotes.",
        "The human body has about 20,000 protein-coding genes, each capable of producing multiple protein variants through alternative splicing.",
        "A codon (3 nucleotides) codes for 1 amino acid. A 300-nucleotide mRNA produces a 100-amino-acid protein.",
    ],
    misconceptions=[
        ("Eating protein directly becomes muscle in your body",
         "Dietary protein is first digested into individual amino acids, which are absorbed into the bloodstream. The body then reassembles these amino acids into its own proteins as needed, using instructions from DNA. You don't absorb chicken muscle and paste it onto your biceps."),
    ],
    analogies=["Protein synthesis is like a factory assembly line — DNA is the master blueprint locked in the office (nucleus), mRNA is a photocopy carried to the factory floor (ribosome), and tRNA workers bring the right parts (amino acids) to build the product (protein)."],
    prerequisites=["DNA structure", "basic cell biology"],
    formulas=["DNA → mRNA (transcription) → Protein (translation)", "3 nucleotides = 1 codon = 1 amino acid"],
    document_evidence=[
        "The text explains: 'Transcription occurs in the nucleus, where RNA polymerase synthesizes mRNA from a DNA template. Translation occurs at ribosomes, where tRNA delivers amino acids matching the mRNA codons.'",
        "The codon table (Figure 5.4) maps each 3-letter codon to its corresponding amino acid or stop signal.",
    ],
    practice_problems=[
        "Given the DNA template strand 3'-TACGGA-5', write the mRNA sequence and the resulting amino acid sequence.",
        "Explain why a mutation in the DNA sequence could result in a non-functional protein.",
    ],
    hints=[
        "mRNA is complementary to the template DNA strand (A→U, T→A, G→C, C→G).",
        "Not all DNA mutations change the protein — the genetic code has redundancy (multiple codons for the same amino acid).",
    ],
    follow_up_questions=[
        "What is the role of tRNA in translation?",
        "How does alternative splicing increase protein diversity?",
    ],
    diagnostic_questions=[
        "Where in the cell does transcription occur? Where does translation occur?",
        "What determines the order of amino acids in a protein?",
    ],
)

_reg("nutrition_food_science", "Vitamin Absorption",
    explanation="the process by which vitamins are taken up from digested food in the small intestine and transported to cells, differing significantly between fat-soluble vitamins (A, D, E, K) and water-soluble vitamins (B-complex, C)",
    reason="proper vitamin absorption is essential for health, and understanding it explains why some vitamins require dietary fat, why deficiencies occur, and how supplements should be taken",
    concrete_examples=[
        "Taking vitamin D with a meal containing fat (like avocado toast) increases absorption because vitamin D is fat-soluble.",
        "Excess water-soluble vitamins (like vitamin C) are excreted in urine, which is why you need them daily.",
        "People who have had gastric bypass surgery often have vitamin absorption problems because the absorptive surface is reduced.",
    ],
    numerical_examples=[
        "The recommended daily intake of vitamin D is 600–800 IU. Absorption increases from ~10% to ~30% when taken with 15g of dietary fat.",
        "Fat-soluble vitamins can accumulate in liver and fat tissue. Vitamin A toxicity can occur above 10,000 IU/day over prolonged periods.",
        "Water-soluble vitamin C has a half-life of about 10–20 days in the body, requiring regular dietary intake.",
    ],
    misconceptions=[
        ("Taking more vitamins is always better for your health",
         "Fat-soluble vitamins (A, D, E, K) can accumulate to toxic levels because they are stored in body fat and liver. Excess vitamin A causes liver damage; excess vitamin D causes calcium buildup. Water-soluble vitamins are generally safer in excess but still have upper limits."),
    ],
    analogies=["Vitamin absorption is like shipping goods — fat-soluble vitamins are like oil paints that need a fat 'container' to travel through the gut, while water-soluble vitamins dissolve freely in the body's water and any excess gets flushed away."],
    prerequisites=["basic biology", "digestive system basics"],
    formulas=["Fat-soluble: A, D, E, K (stored in fat)", "Water-soluble: B-complex, C (excreted in urine)"],
    document_evidence=[
        "The text states: 'Fat-soluble vitamins require bile salts and dietary lipids for absorption in the small intestine.'",
        "Table 4.3 compares the absorption mechanisms, storage sites, and toxicity risks of fat-soluble vs. water-soluble vitamins.",
    ],
    practice_problems=[
        "Explain why a person on a very low-fat diet might develop vitamin D deficiency even with adequate sun exposure.",
        "Which vitamins pose a greater toxicity risk if over-consumed: fat-soluble or water-soluble? Explain why.",
    ],
    hints=[
        "'ADEK' are fat-soluble. Think: 'A DEK of cards needs fat.'",
        "Water-soluble vitamins need daily intake because they aren't stored long-term.",
    ],
    follow_up_questions=[
        "How does gut health affect vitamin absorption?",
        "Why might elderly people have different vitamin absorption than younger adults?",
    ],
    diagnostic_questions=[
        "If you take a vitamin D supplement on an empty stomach, will you absorb as much as with a meal? Why?",
        "Why can't you 'stock up' on vitamin C by taking a large dose once a week?",
    ],
)

_reg("nutrition_food_science", "Food Preservation",
    explanation="the methods used to prevent food spoilage caused by microorganisms, enzymes, and oxidation, thereby extending shelf life and maintaining safety",
    reason="it prevents foodborne illness, reduces food waste, and enables year-round access to seasonal foods",
    concrete_examples=[
        "Refrigeration slows bacterial growth: milk lasts 1–2 weeks refrigerated but spoils in hours at room temperature.",
        "Canning involves heating food to kill bacteria and sealing it in airtight containers — canned vegetables can last 2–5 years.",
        "Salting and smoking fish draws out moisture and creates conditions hostile to bacteria, preserving fish for months.",
    ],
    numerical_examples=[
        "The 'danger zone' for bacterial growth is 40–140°F (4–60°C). At 90°F, bacteria can double every 20 minutes.",
        "Pasteurization heats milk to 161°F (72°C) for 15 seconds, killing 99.999% of harmful bacteria.",
        "Water activity (aw) below 0.85 prevents most bacterial growth. Honey has aw ~0.6, explaining its long shelf life.",
    ],
    misconceptions=[
        ("Freezing food kills all bacteria",
         "Freezing does NOT kill bacteria — it puts them in a dormant state. Once food thaws, bacteria resume multiplying. That's why you should never refreeze thawed raw meat without cooking it first. Proper cooking (not freezing) is what kills most pathogens."),
    ],
    analogies=["Food preservation is like pressing 'pause' on the bacterial clock — refrigeration slows the clock, freezing stops it, but only cooking or sterilization actually destroys the bacteria."],
    prerequisites=["basic biology", "microbiology basics"],
    formulas=["Bacterial doubling: N = N₀ × 2^(t/generation_time)"],
    document_evidence=[
        "The text explains: 'The five main preservation methods are: temperature control, moisture reduction, chemical preservation, irradiation, and modified atmosphere packaging.'",
        "Table 6.2 lists minimum cooking temperatures for different food types to ensure pathogen destruction.",
    ],
    practice_problems=[
        "If bacteria double every 30 minutes at room temperature and you start with 1,000 bacteria, how many will there be after 4 hours?",
        "Explain why beef jerky lasts much longer than fresh beef without refrigeration.",
    ],
    hints=[
        "Bacteria need warmth, moisture, nutrients, and time to grow. Remove any one factor to preserve food.",
        "Water activity is key: reducing available water (drying, salting, adding sugar) inhibits microbial growth.",
    ],
    follow_up_questions=[
        "Why do some preserved foods still need refrigeration after opening?",
        "What is the difference between pasteurization and sterilization?",
    ],
    diagnostic_questions=[
        "If frozen chicken thaws on the counter for 6 hours, is it still safe? Why or why not?",
        "How does salt preserve food?",
    ],
)

_reg("nutrition_food_science", "Enzyme Activity",
    explanation="the catalytic function of enzymes — biological proteins that speed up chemical reactions in living organisms without being consumed, by lowering the activation energy required",
    reason="enzymes control virtually every metabolic process: digestion, energy production, DNA replication, and detoxification",
    concrete_examples=[
        "Amylase in saliva breaks down starch into sugar — that's why bread tastes sweet if you chew it long enough.",
        "Lactase breaks down lactose (milk sugar); people who lack it are lactose intolerant.",
        "Pineapple contains bromelain, a protease enzyme that tenderizes meat by breaking down protein fibers.",
    ],
    numerical_examples=[
        "Catalase decomposes H₂O₂ at a rate of 40 million molecules per second — one of the fastest enzymes known.",
        "Most human enzymes work optimally at 37°C (body temperature) and denature above 40–50°C.",
        "A 10°C increase typically doubles enzymatic reaction rate (Q₁₀ ≈ 2), until the denaturation temperature.",
    ],
    misconceptions=[
        ("Enzymes are consumed in reactions like other reagents",
         "Enzymes are catalysts — they speed up reactions without being used up. After facilitating a reaction, the enzyme returns to its original shape and can catalyze another reaction. A single enzyme molecule can process millions of substrate molecules."),
    ],
    analogies=["An enzyme is like a locksmith's key-cutting machine — it shapes raw keys (substrates) into finished keys (products) over and over without wearing out, making the process much faster than filing by hand."],
    prerequisites=["basic chemistry", "protein structure"],
    formulas=["Rate ∝ [Enzyme] × [Substrate] (simplified)", "Michaelis-Menten: v = Vmax[S] / (Km + [S])"],
    document_evidence=[
        "The text states: 'Enzymes lower activation energy by providing an alternative reaction pathway, binding substrates at their active site.'",
        "Graph 7.1 shows how reaction rate varies with substrate concentration, temperature, and pH.",
    ],
    practice_problems=[
        "Explain what happens to enzyme activity if temperature increases from 37°C to 60°C.",
        "Why does changing the pH dramatically affect enzyme function?",
    ],
    hints=[
        "Enzymes are substrate-specific — the active site fits only certain substrates (lock-and-key model).",
        "Denaturation changes the enzyme's 3D shape, destroying the active site.",
    ],
    follow_up_questions=[
        "What is the difference between competitive and non-competitive enzyme inhibition?",
        "How do cells regulate enzyme activity?",
    ],
    diagnostic_questions=[
        "What happens to an enzyme after it catalyzes a reaction?",
        "Why does cooking an egg change its texture permanently?",
    ],
)

_reg("nutrition_food_science", "Metabolism",
    explanation="the sum of all chemical reactions in a living organism that convert food into energy (catabolism) and use that energy to build molecules (anabolism)",
    reason="it determines how the body processes nutrients, regulates weight, generates energy, and maintains cellular functions",
    concrete_examples=[
        "After eating a sandwich, your metabolism breaks down carbs into glucose (catabolism), then uses that glucose to produce ATP for energy.",
        "Building muscle after exercise involves anabolic metabolism — assembling amino acids into new muscle proteins.",
        "Your basal metabolic rate (BMR) accounts for about 60-75% of daily calorie expenditure, just to maintain basic functions.",
    ],
    numerical_examples=[
        "BMR for a 70 kg male ≈ 10 × 70 + 6.25 × 175 − 5 × 30 + 5 = 1,648 kcal/day (using the Mifflin-St Jeor equation).",
        "1 molecule of glucose through aerobic respiration yields approximately 36-38 ATP molecules.",
        "A person with a TDEE of 2,500 kcal who eats 2,000 kcal/day creates a 500 kcal deficit, losing roughly 1 lb per week.",
    ],
    misconceptions=[
        ("A slow metabolism is the main cause of obesity",
         "Research shows that overweight individuals actually have HIGHER absolute metabolic rates because more mass requires more energy. Obesity is primarily driven by caloric surplus — eating more calories than expended — combined with genetic, behavioral, and environmental factors."),
    ],
    analogies=["Metabolism is like a city's infrastructure — catabolism is the power plants burning fuel (food) to generate electricity (energy), and anabolism is the construction crews using that electricity to build new buildings (cells and tissues)."],
    prerequisites=["basic biology", "chemistry basics"],
    formulas=["BMR (Mifflin-St Jeor): 10×weight(kg) + 6.25×height(cm) − 5×age + s (s=+5 male, −161 female)", "ATP yield from glucose ≈ 36-38 ATP"],
    document_evidence=[
        "The text explains: 'Catabolism breaks down complex molecules into simpler ones, releasing energy. Anabolism uses energy to build complex molecules from simpler ones.'",
        "Figure 8.3 illustrates the relationship between BMR, thermic effect of food, and physical activity in total daily energy expenditure.",
    ],
    practice_problems=[
        "Calculate BMR for a 65 kg, 160 cm, 25-year-old female using the Mifflin-St Jeor equation.",
        "If someone's TDEE is 2,200 kcal and they consume 2,700 kcal daily, how long until they gain 1 lb (assuming 3,500 kcal = 1 lb)?",
    ],
    hints=[
        "Basal metabolic rate is what your body burns at rest — just to breathe, pump blood, and maintain temperature.",
        "Anabolism = building up (needs energy). Catabolism = breaking down (releases energy).",
    ],
    follow_up_questions=[
        "Does exercise increase your resting metabolic rate? If so, how?",
        "What hormones regulate metabolism?",
    ],
    diagnostic_questions=[
        "What is the difference between catabolism and anabolism?",
        "If you eat 500 fewer calories than your body needs daily, what happens over time?",
    ],
)

_reg("nutrition_food_science", "Nutrient Deficiencies",
    explanation="conditions that arise when the body does not receive or absorb adequate amounts of essential nutrients (vitamins, minerals, or macronutrients), leading to specific health problems",
    reason="they cause a wide range of diseases — from scurvy (vitamin C) to anemia (iron) to rickets (vitamin D) — and understanding them is critical for public health and clinical nutrition",
    concrete_examples=[
        "Iron deficiency anemia causes fatigue because hemoglobin (which carries oxygen) requires iron. It's the most common nutritional deficiency worldwide.",
        "Vitamin C deficiency causes scurvy — bleeding gums, bruising, and poor wound healing — as seen historically in sailors without access to fresh fruit.",
        "Vitamin D deficiency leads to rickets in children (soft, weak bones) because vitamin D is needed to absorb calcium.",
    ],
    numerical_examples=[
        "About 2 billion people worldwide are affected by iron deficiency, making it the #1 nutritional deficiency.",
        "The recommended daily intake of vitamin C is 65–90 mg. Scurvy symptoms appear after about 3 months of intake below 10 mg/day.",
        "Severe iodine deficiency during pregnancy can lower a child's IQ by 10–15 points due to impaired thyroid function.",
    ],
    misconceptions=[
        ("Nutrient deficiencies only occur in developing countries",
         "Deficiencies are common worldwide. In developed countries, vitamin D deficiency affects about 40% of adults, iron deficiency affects many women, and B12 deficiency is common among vegetarians. Processed food diets can lead to multiple micronutrient gaps."),
    ],
    analogies=["A nutrient deficiency is like a missing ingredient in a recipe — if you leave out baking powder (vitamin D), the cake won't rise (bones won't strengthen), even if all other ingredients are present."],
    prerequisites=["basic biology", "vitamins and minerals"],
    formulas=["RDA (Recommended Dietary Allowance) varies by nutrient, age, and sex"],
    document_evidence=[
        "The text states: 'Micronutrient deficiencies are often called 'hidden hunger' because symptoms may not appear until the deficiency is severe.'",
        "Table 9.1 links specific deficiencies to their symptoms: iron→anemia, vitamin C→scurvy, vitamin D→rickets, B12→neurological damage.",
    ],
    practice_problems=[
        "A patient presents with fatigue, pale skin, and brittle nails. Which nutrient deficiency is most likely? What dietary changes would you recommend?",
        "Why might a strict vegan be at risk for vitamin B12 deficiency?",
    ],
    hints=[
        "Match the symptom to the nutrient: bone issues → vitamin D/calcium, bleeding → vitamin C/K, nerve issues → B12.",
        "Deficiencies can be caused by inadequate intake, poor absorption, or increased needs (pregnancy, illness).",
    ],
    follow_up_questions=[
        "How can cooking methods affect the nutrient content of food?",
        "What is the difference between a deficiency and a toxicity?",
    ],
    diagnostic_questions=[
        "Which deficiency causes night blindness?",
        "Why might someone eating plenty of food still have a nutrient deficiency?",
    ],
)

_reg("nutrition_food_science", "Food Safety",
    explanation="the practices and procedures that prevent foodborne illness by ensuring food is properly handled, prepared, stored, and served to minimize contamination from pathogens, chemicals, and physical hazards",
    reason="improper food safety causes an estimated 48 million foodborne illnesses and 3,000 deaths annually in the US alone, making it a critical public health concern",
    concrete_examples=[
        "Cross-contamination: using the same cutting board for raw chicken and salad without washing it can transfer Salmonella to the salad.",
        "The '2-hour rule': perishable food left at room temperature for more than 2 hours should be discarded.",
        "Proper handwashing (20 seconds with soap) before handling food is the single most effective food safety practice.",
    ],
    numerical_examples=[
        "Chicken must reach an internal temperature of 165°F (74°C) to kill Salmonella — a food thermometer is essential.",
        "Refrigerators should be at or below 40°F (4°C) and freezers at 0°F (−18°C) to keep food safely stored.",
        "HACCP identifies 7 principles for systematic food safety management, with critical control points monitored at specific temperatures.",
    ],
    misconceptions=[
        ("You can tell if food is safe by smelling or tasting it",
         "Many dangerous pathogens — including Salmonella, E. coli O157:H7, and Listeria — produce no change in food's appearance, smell, or taste. Food can look and smell perfectly fine but contain millions of harmful bacteria. You CANNOT rely on senses to determine food safety."),
    ],
    analogies=["Food safety is like defensive driving — you can't see the danger (bacteria) directly, so you follow systematic rules (temperature control, hygiene, separation) to stay safe."],
    prerequisites=["basic biology", "microbiology basics"],
    formulas=["Danger zone: 40–140°F (4–60°C)", "Minimum cooking temps vary by food type"],
    document_evidence=[
        "The text states: 'The four core principles of food safety are: Clean, Separate, Cook, and Chill.'",
        "Table 10.1 lists minimum internal cooking temperatures: poultry 165°F, ground beef 160°F, whole cuts 145°F.",
    ],
    practice_problems=[
        "A restaurant leaves potato salad on a buffet at 75°F for 3 hours. Is it still safe to eat? Explain.",
        "Design a HACCP plan for a sandwich-making operation. Identify at least 3 critical control points.",
    ],
    hints=[
        "Remember the four Cs: Clean (hands/surfaces), Combat (cross-contamination), Cook (to safe temps), Chill (refrigerate promptly).",
        "When in doubt, throw it out — you can't see or smell most dangerous bacteria.",
    ],
    follow_up_questions=[
        "What is HACCP and why is it used in food manufacturing?",
        "How does refrigeration prevent foodborne illness?",
    ],
    diagnostic_questions=[
        "Can you determine if cooked chicken is safe by cutting it open and looking at the color?",
        "What is cross-contamination and how do you prevent it?",
    ],
)

_reg("nutrition_food_science", "Dietary Fiber",
    explanation="the indigestible plant carbohydrates (soluble and insoluble) that pass through the digestive system largely intact, providing bulk, feeding gut bacteria, and regulating digestion",
    reason="adequate fiber intake reduces risk of heart disease, diabetes, and colon cancer, promotes healthy bowel function, and helps maintain a healthy weight",
    concrete_examples=[
        "Oatmeal is rich in soluble fiber (beta-glucan) which forms a gel in the gut that slows sugar absorption and lowers cholesterol.",
        "Wheat bran contains insoluble fiber that adds bulk to stool and prevents constipation.",
        "Beans and lentils are excellent sources of both soluble and insoluble fiber, providing 15-18g per cup.",
    ],
    numerical_examples=[
        "Recommended daily fiber intake: 25g for women, 38g for men. Average American intake is only about 15g/day.",
        "One cup of lentils provides about 16g of fiber — roughly half the daily recommendation for women.",
        "Increasing fiber from 15g to 30g/day is associated with a 27% lower risk of developing Type 2 diabetes.",
    ],
    misconceptions=[
        ("Fiber supplements are as effective as fiber from whole foods",
         "Whole food sources of fiber come packaged with vitamins, minerals, phytochemicals, and water that supplements lack. Studies show that the health benefits of fiber — like reduced cancer risk and improved gut microbiome diversity — are strongest from whole food sources, not isolated supplements."),
    ],
    analogies=["Dietary fiber is like a broom for your digestive system — soluble fiber acts like a sponge (soaking up cholesterol and sugar), while insoluble fiber acts like bristles (sweeping waste through the intestines)."],
    prerequisites=["basic biology", "carbohydrate chemistry"],
    formulas=["Daily recommended intake: 14g per 1,000 calories consumed"],
    document_evidence=[
        "The text explains: 'Soluble fiber dissolves in water to form a viscous gel that slows digestion, while insoluble fiber adds bulk and accelerates intestinal transit.'",
        "Research summary (Box 11.1): Higher fiber intake is consistently associated with lower rates of cardiovascular disease, type 2 diabetes, and colorectal cancer.",
    ],
    practice_problems=[
        "Plan a day's meals that provide at least 30g of fiber using common foods.",
        "Explain the difference between soluble and insoluble fiber and give two food sources of each.",
    ],
    hints=[
        "Soluble fiber: oats, beans, apples, citrus (dissolves in water, lowers cholesterol).",
        "Insoluble fiber: wheat bran, vegetables, whole grains (doesn't dissolve, prevents constipation).",
    ],
    follow_up_questions=[
        "How does fiber feed beneficial gut bacteria?",
        "Why should you increase fiber intake gradually rather than all at once?",
    ],
    diagnostic_questions=[
        "Why can't humans digest fiber like they digest starch?",
        "What happens in the digestive system when you eat soluble fiber?",
    ],
)

# --- GENERAL ACADEMIC ---

_reg("general_academic", "Critical Thinking",
    explanation="the disciplined process of actively analyzing, evaluating, and synthesizing information to form well-reasoned judgments, rather than accepting claims at face value",
    reason="it is the foundation of academic inquiry, informed decision-making, and protection against manipulation and misinformation",
    concrete_examples=[
        "Reading a news article and checking the primary source rather than trusting the headline is critical thinking.",
        "When a friend says 'everyone is switching to X brand,' questioning whether 'everyone' is an exaggeration uses critical thinking.",
        "Evaluating whether a study's sample size is large enough before accepting its conclusions.",
    ],
    numerical_examples=[
        "A study claims 'Product X reduces cold duration by 50%.' Critical analysis: the reduction was from 6 days to 3 days in a sample of only 20 people — the sample is too small for strong conclusions.",
        "A politician says 'crime has doubled.' Critical check: if crime went from 2 incidents to 4 incidents in a small area, that's technically correct but misleading in context.",
        "An ad says '4 out of 5 dentists recommend' — critical thinking asks: How were the dentists selected? What were they asked?",
    ],
    misconceptions=[
        ("Critical thinking means being negative or cynical about everything",
         "Critical thinking is about being analytically fair, not negative. It means evaluating evidence objectively — which can lead to SUPPORTING a claim as well as rejecting one. A critical thinker follows evidence wherever it leads, whether it confirms or challenges their initial view."),
    ],
    analogies=["Critical thinking is like being a detective — you don't accept the first explanation; you examine evidence, question witnesses, and look for what doesn't fit before drawing conclusions."],
    prerequisites=["basic literacy", "logical reasoning"],
    formulas=["N/A — critical thinking is a process, not a formula"],
    document_evidence=[
        "The text defines: 'Critical thinking involves asking: What is the claim? What is the evidence? Is the reasoning valid? Are there alternative explanations?'",
        "Bloom's Taxonomy (Figure 1.2) places critical thinking skills (analyze, evaluate, create) at the top of the learning hierarchy.",
    ],
    practice_problems=[
        "Evaluate this claim: 'A study of 15 people found that drinking green tea cured their insomnia.' What questions should you ask?",
        "Identify the logical flaw: 'My grandfather smoked all his life and lived to 95, so smoking can't be that bad.'",
    ],
    hints=[
        "Ask: What is the evidence? Is it anecdotal or systematic? Could there be another explanation?",
        "Watch for logical fallacies like hasty generalization, appeal to authority, and false dichotomy.",
    ],
    follow_up_questions=[
        "How can confirmation bias undermine critical thinking?",
        "What is the difference between skepticism and cynicism in critical analysis?",
    ],
    diagnostic_questions=[
        "Is it critical thinking to always disagree with experts?",
        "How would you evaluate a claim that you already agree with?",
    ],
)

_reg("general_academic", "Research Methods",
    explanation="the systematic approaches used to collect, analyze, and interpret data to answer research questions, including experimental, observational, survey, and qualitative methods",
    reason="they ensure findings are reliable, valid, and reproducible, forming the backbone of scientific knowledge and evidence-based policy",
    concrete_examples=[
        "A randomized controlled trial (RCT) testing a new drug randomly assigns patients to treatment or placebo groups to control for confounding factors.",
        "A survey of 1,000 voters about their preferred candidate uses random sampling to estimate the views of the entire electorate.",
        "An ethnographic study where a researcher lives with a community for a year to understand their cultural practices uses qualitative methods.",
    ],
    numerical_examples=[
        "A study with n=30 has a larger standard error than one with n=3,000. Doubling sample size reduces standard error by a factor of √2 ≈ 1.41.",
        "If a clinical trial shows p=0.03, there is a 3% probability of seeing results this extreme if the drug had no effect.",
        "An effect size of d=0.8 is considered large, d=0.5 medium, and d=0.2 small in social science research.",
    ],
    misconceptions=[
        ("Correlation implies causation",
         "A correlation between two variables only means they move together; it does NOT mean one causes the other. Ice cream sales and drowning rates are correlated (both increase in summer), but ice cream doesn't cause drowning. Only controlled experiments or very careful causal analysis can establish causation."),
    ],
    analogies=["Research methods are like cooking recipes — following them carefully ensures your results (dish) turn out as expected. Skipping steps or substituting ingredients (cutting corners) leads to unreliable results."],
    prerequisites=["basic statistics", "critical thinking"],
    formulas=["Standard error = SD / √n", "Effect size (Cohen's d) = (M₁ − M₂) / SDpooled"],
    document_evidence=[
        "The text explains: 'The gold standard for establishing causation is the randomized controlled trial, where random assignment ensures groups are comparable before treatment.'",
        "Table 2.1 compares experimental, quasi-experimental, correlational, and descriptive research designs.",
    ],
    practice_problems=[
        "A study finds that students who eat breakfast score higher on tests. Can we conclude that eating breakfast causes better scores? What would a better study design look like?",
        "What is the purpose of a control group in an experiment?",
    ],
    hints=[
        "Randomization controls for confounding variables — factors other than the treatment that might affect the outcome.",
        "Internal validity = can we trust the causal claim? External validity = can we generalize to other populations?",
    ],
    follow_up_questions=[
        "What is the difference between reliability and validity in research?",
        "When would qualitative methods be more appropriate than quantitative methods?",
    ],
    diagnostic_questions=[
        "What makes an experiment 'controlled'?",
        "If a study finds a correlation of r=0.85 between hours studied and exam score, does studying cause better scores?",
    ],
)

_reg("general_academic", "Argument Structure",
    explanation="the logical framework of a reasoned argument, consisting of premises (supporting statements), conclusions (the claim being supported), and the logical connections between them",
    reason="understanding argument structure is essential for constructing persuasive essays, evaluating others' reasoning, and identifying fallacious logic",
    concrete_examples=[
        "Premise 1: All mammals are warm-blooded. Premise 2: Dolphins are mammals. Conclusion: Dolphins are warm-blooded. (Valid deductive argument.)",
        "A student arguing 'Universities should allow pets in dorms because research shows pets reduce stress' needs evidence linking that research to the specific dorm context.",
        "An editorial arguing 'we should ban plastic bags' must provide premises about environmental harm, available alternatives, and feasibility.",
    ],
    numerical_examples=[
        "In a formal debate, a team might present 3 main claims, each supported by 2-3 pieces of evidence, for 6-9 total evidence citations.",
        "If 80% of peer-reviewed studies support conclusion X, the argument for X is stronger than if only 20% support it.",
        "A survey of 100 arguments found 47% contained at least one logical fallacy — nearly half of everyday reasoning is flawed.",
    ],
    misconceptions=[
        ("An argument with true premises always has a true conclusion",
         "Only VALID deductive arguments guarantee a true conclusion from true premises. Invalid arguments can have true premises but a false conclusion. Inductive arguments, even with strong premises, only make the conclusion probable, not certain. The structure of the argument matters as much as the truth of the premises."),
    ],
    analogies=["An argument is like a building — premises are the foundation and walls (support), the conclusion is the roof (what everything holds up), and the logic is the architecture (how it all fits together)."],
    prerequisites=["basic literacy", "logical reasoning"],
    formulas=["Deductive: If P₁ and P₂, then C (conclusion follows necessarily)", "Inductive: P₁, P₂, ... → C is probable"],
    document_evidence=[
        "The text states: 'A sound argument has two properties: it is valid (the conclusion follows logically from the premises) and its premises are all true.'",
        "Section 3.2 distinguishes between deductive arguments (certainty) and inductive arguments (probability).",
    ],
    practice_problems=[
        "Identify the premises and conclusion: 'Students who study regularly perform better on exams. You should study regularly because you want to perform well.'",
        "Is this argument valid? 'All birds can fly. Penguins are birds. Therefore, penguins can fly.' Explain.",
    ],
    hints=[
        "Look for conclusion indicators: therefore, so, thus, consequently, it follows that.",
        "Look for premise indicators: because, since, given that, as evidenced by.",
    ],
    follow_up_questions=[
        "What is the difference between a valid argument and a sound argument?",
        "How can an argument be valid but not sound?",
    ],
    diagnostic_questions=[
        "What is a premise and how does it differ from a conclusion?",
        "Can a false conclusion come from true premises in a valid argument?",
    ],
)

_reg("general_academic", "Evidence Evaluation",
    explanation="the process of assessing the quality, relevance, and strength of evidence used to support claims, considering source reliability, methodology, sample size, and potential biases",
    reason="it protects against misinformation and enables informed decisions by distinguishing strong evidence from weak, biased, or fabricated claims",
    concrete_examples=[
        "A peer-reviewed medical study is stronger evidence than a single doctor's opinion because it has been vetted by other experts.",
        "A product review from a verified purchaser is more reliable than a review from an anonymous account with no purchase history.",
        "Government statistics from census data are generally more reliable than self-reported online surveys.",
    ],
    numerical_examples=[
        "A meta-analysis combining 50 studies with a total n=15,000 provides stronger evidence than any single study alone.",
        "A study with 95% confidence level and p=0.01 provides stronger evidence against the null hypothesis than one with p=0.04.",
        "If a study has an attrition rate of 40% (many participants dropped out), its results are less trustworthy.",
    ],
    misconceptions=[
        ("Published research is always correct and unbiased",
         "Publication bias, funding conflicts, methodological flaws, and even fraud affect published research. Studies funded by companies are more likely to report favorable results. Replication crises in psychology and medicine show that many published findings fail to reproduce. Always evaluate methodology, not just the journal name."),
    ],
    analogies=["Evaluating evidence is like being a judge in a courtroom — you must weigh the testimony (evidence) by considering the witness's (source's) credibility, consistency, and potential motives before reaching a verdict."],
    prerequisites=["critical thinking", "basic statistics"],
    formulas=["Hierarchy of evidence: Systematic reviews > RCTs > Cohort studies > Case studies > Expert opinion"],
    document_evidence=[
        "The text presents the evidence pyramid: 'Systematic reviews and meta-analyses represent the strongest form of evidence, while anecdotal reports represent the weakest.'",
        "Section 4.3 discusses common biases: confirmation bias, publication bias, selection bias, and recall bias.",
    ],
    practice_problems=[
        "Rank these in order of evidence strength: a celebrity endorsement, a randomized controlled trial, a case study, a meta-analysis.",
        "A pharmaceutical company funds a study of its own drug and reports positive results. What concerns should you have?",
    ],
    hints=[
        "Ask: Who conducted the study? Who funded it? Was there a control group? How large was the sample?",
        "Replication is key — a finding is more trustworthy if multiple independent groups have confirmed it.",
    ],
    follow_up_questions=[
        "What is the 'replication crisis' and why does it matter?",
        "How does peer review work, and what are its limitations?",
    ],
    diagnostic_questions=[
        "Is a study published in a prestigious journal automatically good evidence?",
        "What makes anecdotal evidence less reliable than statistical evidence?",
    ],
)

_reg("general_academic", "Logical Fallacies",
    explanation="errors in reasoning that undermine the logic of an argument, making the conclusion unreliable even if it sounds persuasive",
    reason="recognizing fallacies is essential for evaluating arguments, avoiding manipulation, and constructing sound reasoning in academic and everyday contexts",
    concrete_examples=[
        "Ad hominem: 'You can't trust the professor's climate research because she drives an SUV.' (Attacks the person, not the argument.)",
        "Straw man: 'My opponent wants to reduce military spending, so they clearly want to leave us defenseless.' (Misrepresents the position.)",
        "False dichotomy: 'You're either with us or against us.' (Ignores middle ground and other options.)",
    ],
    numerical_examples=[
        "Hasty generalization: 'I surveyed 3 people and all 3 support Policy X, so 100% of people must support it.' (Sample far too small.)",
        "Gambler's fallacy: 'The roulette wheel landed on red 5 times, so black is due.' (Each spin is independent; P(black) is still ~47%.)",
        "Base rate fallacy: A test with 99% accuracy applied to a population where only 1% are positive means most positives are false positives.",
    ],
    misconceptions=[
        ("If someone uses a fallacy, their conclusion must be wrong",
         "A fallacious argument means the reasoning is flawed, not that the conclusion is necessarily false. The conclusion might still be true — it's just not properly supported by the given argument. This is called the 'fallacy fallacy' — concluding something is false BECAUSE the argument for it contains a fallacy."),
    ],
    analogies=["Logical fallacies are like optical illusions for the mind — they look convincing at first glance but fall apart under careful examination."],
    prerequisites=["basic literacy", "logical reasoning"],
    formulas=["No formula — fallacy identification requires recognizing patterns in reasoning"],
    document_evidence=[
        "The text lists 15 common fallacies: ad hominem, straw man, false dichotomy, appeal to authority, red herring, slippery slope, circular reasoning, hasty generalization, post hoc, tu quoque, bandwagon, false cause, appeal to emotion, equivocation, and burden of proof reversal.",
        "Section 5.2 provides real-world examples of each fallacy from media and political discourse.",
    ],
    practice_problems=[
        "Identify the fallacy: 'If we allow students to use calculators on tests, next they'll want to use AI to write all their papers.' (Slippery slope)",
        "Identify the fallacy: 'This diet must work because a famous actress uses it.' (Appeal to authority/celebrity)",
    ],
    hints=[
        "Ask: Does the argument actually support the conclusion, or does it distract, mislead, or oversimplify?",
        "Common patterns: attacking the person (ad hominem), misrepresenting the argument (straw man), offering only two options (false dichotomy).",
    ],
    follow_up_questions=[
        "Can an argument contain a fallacy and still reach a correct conclusion?",
        "How can you politely point out a fallacy in someone's argument?",
    ],
    diagnostic_questions=[
        "What is the difference between ad hominem and a legitimate criticism of someone's expertise?",
        "Give an example of the post hoc fallacy from everyday life.",
    ],
)

_reg("general_academic", "Study Design",
    explanation="the plan and structure of a research study that determines how data will be collected, what comparisons will be made, and how variables will be controlled to answer a research question",
    reason="good study design minimizes bias, maximizes validity, and ensures that conclusions drawn from data are trustworthy and meaningful",
    concrete_examples=[
        "A randomized controlled trial (RCT) randomly assigns participants to treatment or control groups to test a drug's effectiveness.",
        "A longitudinal study follows the same group of people for 20 years to study how diet affects heart disease risk.",
        "A cross-sectional study surveys 5,000 people at a single point in time to estimate the prevalence of depression.",
    ],
    numerical_examples=[
        "Power analysis: To detect a medium effect (d=0.5) with 80% power at α=0.05, you need approximately 64 participants per group.",
        "A study with n=10 per group has very low power (~17%) to detect a small effect, meaning it will likely miss real differences.",
        "Stratified sampling: if 40% of the population is under 30, the sample should also be ~40% under 30.",
    ],
    misconceptions=[
        ("More data always means a better study",
         "Data quality matters more than quantity. A well-designed study with 200 carefully selected participants can be more informative than a poorly designed study with 10,000 participants affected by selection bias, measurement error, or confounding variables."),
    ],
    analogies=["Study design is like building a house's blueprint — you plan every room (variable), doorway (measurement), and structural support (control) BEFORE construction (data collection) begins."],
    prerequisites=["basic statistics", "research methods"],
    formulas=["Power = P(reject H₀ | H₁ true)", "Sample size calculations depend on effect size, power, and α"],
    document_evidence=[
        "The text states: 'The choice of study design should be guided by the research question, ethical constraints, available resources, and the type of evidence needed.'",
        "Figure 6.1 compares the strengths and weaknesses of experimental, quasi-experimental, and observational designs.",
    ],
    practice_problems=[
        "Design a study to test whether meditation reduces exam anxiety in college students. Specify the design type, variables, control group, and outcome measure.",
        "Why would a researcher choose an observational study over an experiment to study the effects of smoking?",
    ],
    hints=[
        "Match your design to your question: causal questions need experiments; prevalence questions need surveys.",
        "Always identify: independent variable, dependent variable, confounding variables, and control conditions.",
    ],
    follow_up_questions=[
        "What is the difference between a between-subjects and within-subjects design?",
        "How does blinding improve the validity of a study?",
    ],
    diagnostic_questions=[
        "What makes a randomized controlled trial the 'gold standard'?",
        "What is a confounding variable and why is it a problem?",
    ],
)

_reg("general_academic", "Bias Identification",
    explanation="the process of recognizing systematic errors in thinking, research, or data collection that skew results or conclusions away from the truth",
    reason="unidentified biases lead to flawed conclusions, unfair decisions, and flawed research — recognizing them is essential for objective analysis",
    concrete_examples=[
        "Confirmation bias: a researcher who believes a treatment works may unconsciously interpret ambiguous results as supportive.",
        "Selection bias: a health survey conducted only in gyms would overestimate the general population's fitness levels.",
        "Survivorship bias: studying only successful companies to find success factors ignores the many failed companies that used the same strategies.",
    ],
    numerical_examples=[
        "In one study, researchers who knew which group received the drug rated symptom improvement 30% higher than blinded researchers — observer bias in action.",
        "An online poll about internet access has 100% self-selection bias: it can only reach people who already have internet access.",
        "Anchoring bias: when asked 'Is the population of Turkey greater or less than 35 million?' people systematically estimate lower than when asked about 85 million.",
    ],
    misconceptions=[
        ("Being aware of your biases is enough to eliminate them",
         "Research shows that simply knowing about a bias does NOT prevent it from affecting your thinking. Deliberate, systematic corrective procedures — like blind review, structured decision-making, pre-registration, and checklists — are needed to mitigate bias effectively."),
    ],
    analogies=["Bias is like a tilt in a pinball machine — the ball (your thinking) always drifts in one direction unless you recognize the tilt and actively correct for it."],
    prerequisites=["critical thinking"],
    formulas=["No formula — bias identification requires awareness and systematic checking"],
    document_evidence=[
        "The text identifies major cognitive biases: 'Confirmation bias, anchoring, availability heuristic, and dunning-kruger effect are among the most impactful biases affecting academic and professional work.'",
        "Section 7.3 describes debiasing strategies: blind review, pre-registration, adversarial collaboration, and red teams.",
    ],
    practice_problems=[
        "A company surveys its current customers to ask if they like the product. What type of bias is present?",
        "A researcher expects her new teaching method to improve test scores. How could this expectation bias her study? What safeguards should she implement?",
    ],
    hints=[
        "Ask: Who was included in the sample? Who was excluded? Could the method systematically favor certain results?",
        "Blinding (not knowing group assignment) is the primary defense against observer bias.",
    ],
    follow_up_questions=[
        "How does pre-registration help reduce researcher bias?",
        "What is the difference between bias and random error?",
    ],
    diagnostic_questions=[
        "What is survivorship bias and how does it mislead conclusions?",
        "If you only read news sources you agree with, what bias are you exhibiting?",
    ],
)

_reg("general_academic", "Effective Communication",
    explanation="the skill of conveying ideas clearly, persuasively, and appropriately to a specific audience through written, spoken, or visual means",
    reason="it is fundamental to academic success, professional achievement, and personal relationships — even brilliant ideas fail if poorly communicated",
    concrete_examples=[
        "Structuring an essay with introduction, body, and conclusion helps readers follow your argument logically.",
        "Using visual aids in a presentation helps the audience grasp complex data that would be hard to follow verbally.",
        "Adjusting your vocabulary and tone when explaining physics to a child vs. a colleague demonstrates audience awareness.",
    ],
    numerical_examples=[
        "Studies show audiences retain about 10% of what they hear after 72 hours, but 65% when visuals are combined with verbal presentation.",
        "The average attention span during a lecture is about 10–20 minutes; breaking content into segments with activities improves engagement by up to 50%.",
        "Emails with clear subject lines are 26% more likely to be opened than those with vague or absent subjects.",
    ],
    misconceptions=[
        ("Good communication means using complex vocabulary and long sentences",
         "Clear communication uses the simplest language that accurately conveys the meaning. Research shows readability improves at 8th-grade reading level. Hemingway, Einstein, and Feynman are celebrated for explaining complex ideas simply. Jargon should be used only when the audience expects it."),
    ],
    analogies=["Communication is like a bridge — it only works if it connects YOUR side (your idea) to THEIR side (their understanding). A beautiful bridge that doesn't reach the other bank is useless."],
    prerequisites=["basic literacy"],
    formulas=["N/A — communication is a skill, not a formula"],
    document_evidence=[
        "The text advises: 'Know your audience, organize your message, use clear language, provide evidence, and seek feedback.'",
        "Research by Mehrabian (cited in Section 8.2) found that nonverbal cues account for a significant portion of face-to-face communication impact.",
    ],
    practice_problems=[
        "Rewrite this sentence for a general audience: 'The pharmacokinetic profile of the compound demonstrated a Cmax of 250 ng/mL with a Tmax of 2 hours.'",
        "You need to give a 5-minute presentation on climate change. Outline your key points for a non-scientific audience.",
    ],
    hints=[
        "Start with the main point, not the background. Audiences want to know WHY before HOW.",
        "Use the 'so what?' test: after every claim, ask if the audience understands why it matters to them.",
    ],
    follow_up_questions=[
        "How does audience analysis change the way you present the same information?",
        "What role does active listening play in effective communication?",
    ],
    diagnostic_questions=[
        "What makes a piece of writing 'clear'?",
        "Why might a brilliant scientist be a poor communicator?",
    ],
)

_reg("general_academic", "Learning Strategies",
    explanation="evidence-based techniques for acquiring, retaining, and applying knowledge more effectively, including spaced repetition, active recall, interleaving, and elaboration",
    reason="using proven learning strategies can dramatically improve academic performance and reduce study time compared to passive review methods",
    concrete_examples=[
        "Spaced repetition: reviewing flashcards at increasing intervals (1 day, 3 days, 7 days, 14 days) rather than cramming the night before.",
        "Active recall: closing the textbook and trying to write down everything you remember, rather than re-reading highlighted passages.",
        "Interleaving: mixing practice problems from different chapters instead of completing all Chapter 5 problems before moving to Chapter 6.",
    ],
    numerical_examples=[
        "Ebbinghaus's forgetting curve: without review, students forget about 70% of material within 24 hours. With one review at 24 hours, retention at 1 week improves from ~30% to ~70%.",
        "A meta-analysis of 10 studies found that practice testing (active recall) improved exam performance by 0.5 standard deviations compared to re-reading.",
        "Students using spaced repetition retained 90% of vocabulary after 6 months vs. 30% for massed practice (cramming).",
    ],
    misconceptions=[
        ("Highlighting and re-reading are effective study methods",
         "Research consistently shows that passive review methods like highlighting and re-reading are among the LEAST effective learning strategies. They create an 'illusion of fluency' — the material feels familiar, but you can't actually recall it on a test. Active recall and practice testing are far more effective."),
    ],
    analogies=["Learning strategies are like workout plans for your brain — just as doing the same easy exercise won't build muscle, re-reading the same notes won't build knowledge. You need to challenge yourself (test yourself) and allow recovery time (spaced repetition)."],
    prerequisites=["basic literacy"],
    formulas=["Spacing effect: optimal review intervals increase over time (e.g., 1, 3, 7, 14, 30 days)"],
    document_evidence=[
        "The text states: 'The most effective learning strategies — practice testing, spaced repetition, and interleaving — are also the ones students use least, preferring easier but less effective methods like re-reading.'",
        "Dunlosky et al. (2013) rated 10 common learning techniques (Table 9.1): practice testing and spaced repetition rated 'high utility'; highlighting and re-reading rated 'low utility.'",
    ],
    practice_problems=[
        "Design a study plan using spaced repetition for an exam 3 weeks away, covering 5 chapters.",
        "Explain why interleaving problems from different topics improves learning, even though it feels harder in the moment.",
    ],
    hints=[
        "If studying feels easy, you're probably not learning much. Desirable difficulties improve long-term retention.",
        "Test yourself BEFORE reviewing — even getting things wrong improves subsequent learning.",
    ],
    follow_up_questions=[
        "What is the 'testing effect' and how does it work?",
        "How can you apply elaboration to deepen understanding of a new concept?",
    ],
    diagnostic_questions=[
        "Why does re-reading notes feel effective even though it isn't?",
        "What is spaced repetition and why does it work better than cramming?",
    ],
)

_reg("general_academic", "Information Synthesis",
    explanation="the process of combining information from multiple sources to create a new, integrated understanding or argument that is more than the sum of its parts",
    reason="it enables original thinking, comprehensive literature reviews, and the ability to connect ideas across disciplines — essential for advanced academic work",
    concrete_examples=[
        "A literature review synthesizes findings from 30 studies on sleep and cognition to identify overall patterns and gaps.",
        "A policy report combines economic data, public health statistics, and case studies to recommend a course of action.",
        "A student connects ideas from biology (evolution), economics (market competition), and psychology (behavioral adaptation) to write a cross-disciplinary essay.",
    ],
    numerical_examples=[
        "A meta-analysis synthesizes effect sizes from 25 studies: individual studies range from d=0.2 to d=0.8, but the pooled effect is d=0.45 with narrow confidence interval.",
        "A systematic review screens 500 papers, includes 47 that meet criteria, and identifies 3 consistent themes across all 47.",
        "A research proposal cites 15 sources to build a theoretical framework, connecting findings from 4 different fields.",
    ],
    misconceptions=[
        ("Synthesis is the same as summarizing multiple sources",
         "Summarizing restates what each source says independently. Synthesis integrates ideas, identifies patterns across sources, resolves contradictions, and creates new understanding. A summary says 'Study A found X; Study B found Y.' A synthesis says 'Studies A and B together suggest Z, but disagree on W, pointing to a need for further research on Q.'"),
    ],
    analogies=["Synthesis is like cooking a dish from multiple ingredients — you don't just pile raw vegetables on a plate (summary). You combine, season, and cook them into something new (synthesis) that wouldn't exist from any single ingredient alone."],
    prerequisites=["critical thinking", "evidence evaluation"],
    formulas=["N/A — synthesis is a skill requiring integration, not computation"],
    document_evidence=[
        "The text explains: 'Effective synthesis goes beyond juxtaposing sources; it identifies agreements, contradictions, and gaps to build a coherent argument or narrative.'",
        "Section 10.3 presents the 'synthesis matrix' technique for organizing findings from multiple sources by theme.",
    ],
    practice_problems=[
        "Given three sources with different conclusions about remote work productivity, write a synthesis paragraph that integrates all three.",
        "Create a synthesis matrix for 4 articles on the topic of social media and mental health.",
    ],
    hints=[
        "Organize by THEME, not by source. Don't summarize Source 1, then Source 2. Instead, discuss Theme A using evidence from Sources 1, 2, and 3.",
        "Look for where sources agree (convergence), disagree (contradiction), or leave questions unanswered (gaps).",
    ],
    follow_up_questions=[
        "How does synthesis differ from analysis?",
        "What is a synthesis matrix and how does it help organize research?",
    ],
    diagnostic_questions=[
        "If you write one paragraph per source, are you synthesizing or summarizing?",
        "What makes a literature review more than just a collection of summaries?",
    ],
)

# ============================================================
# DOMAIN → CONCEPTS LOOKUP
# ============================================================

def get_domain_concepts(domain: str) -> List[str]:
    return [k[1] for k in CONCEPT_REGISTRY if k[0] == domain]

def get_concept_info(domain: str, concept: str) -> Dict[str, Any]:
    return CONCEPT_REGISTRY[(domain, concept)]

# ============================================================
# Now the actual dataset generation script is very large.
# We break it into a separate file to keep this content registry
# manageable. The generation logic imports from here.
# ============================================================

if __name__ == "__main__":
    # Quick verification
    for domain in VALID_DOMAINS:
        concepts = get_domain_concepts(domain)
        print(f"{domain}: {len(concepts)} concepts")
        for c in concepts:
            info = get_concept_info(domain, c)
            assert len(info["concrete_examples"]) >= 3, f"Need ≥3 concrete_examples for {domain}/{c}"
            assert len(info["numerical_examples"]) >= 3, f"Need ≥3 numerical_examples for {domain}/{c}"
            assert len(info["misconceptions"]) >= 1, f"Need ≥1 misconception for {domain}/{c}"
            assert len(info["analogies"]) >= 1, f"Need ≥1 analogy for {domain}/{c}"
            assert len(info["practice_problems"]) >= 2, f"Need ≥2 practice_problems for {domain}/{c}"
            assert len(info["hints"]) >= 2, f"Need ≥2 hints for {domain}/{c}"
            assert len(info["follow_up_questions"]) >= 2, f"Need ≥2 follow_up_questions for {domain}/{c}"
            assert len(info["diagnostic_questions"]) >= 2, f"Need ≥2 diagnostic_questions for {domain}/{c}"
            assert len(info["document_evidence"]) >= 2, f"Need ≥2 document_evidence for {domain}/{c}"
    print(f"\nTotal concepts registered: {len(CONCEPT_REGISTRY)}")
    print("All concept content verified: PASS")
