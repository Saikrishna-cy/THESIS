"""
Controlled Interview Dataset Generator v2
==========================================
Thesis: Hallucination Detection in RAG-Based Systems for Bilingual Interview Transcripts
Leiden University · 2025-2026

Design:
  - 600 students total: 300 English + 300 Dutch
  - Each student = 1 interview = 14 turns (7 interviewer Q + 7 student responses)
  - ONE question per topic — the interview is the SOURCE, not a test instrument
  - Topic order is RANDOMISED per student (harder for LLMs, more realistic)
  - Each student has a consistent identity: name, year, programme, nationality
  - Each student mentions one named third party (advisor / coordinator / staff)

  After the interview exists, 7 RAG queries are run against it:
    5 standard types: participant_content, factual_summary, specific_content,
                      sentiment, temporal
    + refusal_hallucination : targets something EXPLICITLY stated in the interview
    + role_drift            : asks for multi-entity summary to trigger positional
                              speaker decay (Shi et al. 2023 Lost in the Middle)
  refusal and role_drift queries ROTATE across topics per student (full coverage)
  Sentiment: lexicon-based per utterance + per topic + interview level (-1.0 to +1.0)

Output: data/controlled/
  controlled_interviews.json     ← 600 full interview records
  question_bank.json             ← interview questions + 7 RAG query templates
  student_profiles.json          ← all 600 student identity cards
  generation_log.json            ← per-interview status, timing, sentiment stats

Usage:
  python3 generate_controlled_dataset_v3.py --dry-run
  python3 generate_controlled_dataset_v3.py --test-sentiment
  python3 generate_controlled_dataset_v3.py                     # full 600
  python3 generate_controlled_dataset_v3.py --lang en --n 10    # 10 EN only
  python3 generate_controlled_dataset_v3.py --resume            # resume from checkpoint
"""

import os, json, re, uuid, time, argparse, random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — STUDENT IDENTITY POOL
# Realistic diverse names for international (EN) and domestic Dutch (NL) students
# Each student gets: name, year, programme, nationality, personal_detail, third_party
# ─────────────────────────────────────────────────────────────────────────────

PROGRAMMES_EN = [
    "BSc Computer Science", "BSc Psychology", "BSc International Relations",
    "BSc Economics", "MSc Data Science", "MSc Linguistics",
    "BSc Biology", "MSc Public Administration", "BSc History",
    "BSc Law", "MSc Artificial Intelligence", "BSc Media Studies"
]

PROGRAMMES_NL = [
    "BSc Informatica", "BSc Psychologie", "BSc Internationale Betrekkingen",
    "BSc Economie", "MSc Data Science", "BSc Taalkunde",
    "BSc Biologie", "MSc Bestuurskunde", "BSc Geschiedenis",
    "BSc Rechtsgeleerdheid", "MSc Kunstmatige Intelligentie", "BSc Communicatiewetenschappen"
]

# International student names (EN interviews — studying in English at Dutch university)
EN_FIRST_NAMES = [
    "Sai","Krishna","Suresh","Priya","Ananya","Rahul","Divya","Arjun",
    "Wei","Jing","Xiao","Ming","Yuki","Haruto","Sora","Aiko",
    "Fatima","Omar","Layla","Tariq","Amira","Hassan","Yasmin","Karim",
    "Andrei","Ioana","Bogdan","Mirela","Lukas","Petra","Tomasz","Zara",
    "Kwame","Amara","Chioma","Emeka","Abena","Kofi","Ngozi","Seun",
    "Sofia","Mateo","Isabel","Diego","Valentina","Lucas","Camila","Thiago",
    "Emma","Liam","Noah","Olivia","James","Charlotte","Benjamin","Mia"
]

EN_LAST_NAMES = [
    "Krishnamurthy","Patel","Sharma","Nair","Iyer","Reddy","Singh","Gupta",
    "Zhang","Wang","Liu","Chen","Tanaka","Yamamoto","Sato","Kobayashi",
    "Al-Rashid","Hassan","Mansour","Ibrahim","Khalil","Yousef","Nasser","Aziz",
    "Popescu","Ionescu","Dinu","Munteanu","Novak","Horak","Kowalski","Wiśniewska",
    "Mensah","Asante","Osei","Addo","Owusu","Boateng","Acheampong","Amoah",
    "Rodriguez","Garcia","Martinez","Lopez","Ferreira","Oliveira","Santos","Costa",
    "Johnson","Williams","Brown","Taylor","Anderson","Wilson","Moore","Jackson"
]

# Dutch domestic student names (NL interviews)
NL_FIRST_NAMES = [
    "Sven","Lars","Daan","Finn","Bram","Luuk","Tim","Joris",
    "Emma","Sophie","Lisa","Anna","Fleur","Inge","Noor","Roos",
    "Thijs","Niels","Jesse","Ruben","Milan","Robin","Stijn","Bart",
    "Lotte","Eva","Fien","Merel","Iris","Jade","Maud","Vera",
    "Pieter","Thomas","Maarten","Simon","Frank","Kevin","Rick","Mark",
    "Laura","Marieke","Hanneke","Annelies","Joke","Sandra","Ellen","Rianne"
]

NL_LAST_NAMES = [
    "de Vries","van den Berg","van Dijk","Bakker","Janssen","Visser","Smit","Meijer",
    "de Boer","Mulder","de Groot","Bos","Vos","Peters","Hendriks","van Leeuwen",
    "Dekker","Brouwer","de Wit","Dijkstra","Smits","Vermeer","van der Laan","Kuiper",
    "Wolff","Hermans","Lammers","Hoekstra","Schouten","van Beek","Kok","Prins",
    "Willems","Jacobs","van den Heuvel","Bogaard","Scholten","van der Berg","Postma","Groen"
]

PERSONAL_DETAILS_EN = [
    "lives in student housing on Rapenburg","commutes from Rotterdam by train",
    "shares a flat near the university with two other students",
    "lives in an international student residence on Kaiserstraat",
    "rents a room in the city centre","stays in university dormitory block C",
    "commutes from Amsterdam twice a week","lives with a host family in Oegstgeest",
    "recently moved from the student hotel to a private room","lives near Centraal Station"
]

PERSONAL_DETAILS_NL = [
    "woont op de Haarlemmerstraat","pendelt vanuit Rotterdam",
    "deelt een appartement met twee studiegenoten bij de campus",
    "woont in de internationale studentenwoning aan de Kaiserstraat",
    "huurt een kamer in het stadscentrum","verblijft in studentenflat blok C",
    "pendelt twee keer per week vanuit Amsterdam","woont bij een gastgezin in Oegstgeest",
    "verhuisde onlangs van het studentenhotel naar een eigen kamer","woont bij het Centraal Station"
]

# Named third parties — realistic university staff
THIRD_PARTIES_EN = [
    {"name": "Dr. Pieterse",    "role": "academic advisor"},
    {"name": "Ms. van der Berg","role": "student counsellor"},
    {"name": "Mr. Hendriks",    "role": "department coordinator"},
    {"name": "Dr. Janssen",     "role": "study advisor"},
    {"name": "Ms. Bakker",      "role": "financial aid officer"},
    {"name": "Mr. de Vries",    "role": "international student coordinator"},
    {"name": "Dr. Smits",       "role": "course director"},
    {"name": "Ms. Visser",      "role": "housing office manager"},
    {"name": "Mr. Kok",         "role": "IT helpdesk supervisor"},
    {"name": "Dr. Willems",     "role": "faculty administrator"},
]

THIRD_PARTIES_NL = [
    {"name": "dr. Pieterse",       "role": "studieadviseur"},
    {"name": "mevrouw Van der Berg","role": "studentenbegeleider"},
    {"name": "de heer Hendriks",   "role": "onderwijscoördinator"},
    {"name": "dr. Janssen",        "role": "studiecoach"},
    {"name": "mevrouw Bakker",     "role": "financieel adviseur"},
    {"name": "de heer De Vries",   "role": "internationaal studentencoördinator"},
    {"name": "dr. Smits",          "role": "vakcoördinator"},
    {"name": "mevrouw Visser",     "role": "beheerder studentenhuisvesting"},
    {"name": "de heer Kok",        "role": "IT-helpdeskmedewerker"},
    {"name": "dr. Willems",        "role": "faculteitsadministrateur"},
]


def generate_student_profile(language: str, student_idx: int, rng: random.Random) -> dict:
    """Generate a consistent student identity for one interview."""
    if language == "en":
        first = rng.choice(EN_FIRST_NAMES)
        last  = rng.choice(EN_LAST_NAMES)
        prog  = rng.choice(PROGRAMMES_EN)
        detail = rng.choice(PERSONAL_DETAILS_EN)
        third  = rng.choice(THIRD_PARTIES_EN)
        nationalities = ["Indian","Chinese","Nigerian","Romanian","Brazilian",
                         "Egyptian","South Korean","Indonesian","Mexican","Ghanaian"]
        nationality = rng.choice(nationalities)
    else:
        first = rng.choice(NL_FIRST_NAMES)
        last  = rng.choice(NL_LAST_NAMES)
        prog  = rng.choice(PROGRAMMES_NL)
        detail = rng.choice(PERSONAL_DETAILS_NL)
        third  = rng.choice(THIRD_PARTIES_NL)
        nationality = "Dutch"

    year_map = {1: "1st year", 2: "2nd year", 3: "3rd year", 4: "4th year"}
    year_map_nl = {1: "eerstejaars", 2: "tweedejaars", 3: "derdejaars", 4: "vierdejaars"}
    year_num = rng.randint(1, 4)
    year_label = year_map[year_num] if language == "en" else year_map_nl[year_num]

    return {
        "student_id":     f"ctrl_{language}_{str(student_idx).zfill(3)}",
        "language":       language,
        "name":           f"{first} {last}",
        "first_name":     first,
        "year":           year_label,
        "year_num":       year_num,
        "programme":      prog,
        "nationality":    nationality,
        "personal_detail": detail,
        "third_party":    third,   # {name, role}
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — QUESTION BANK
# 5 query types × 7 topics × 2 languages = 70 questions
# Fixed order — every student gets questions in the same type sequence per topic
# ─────────────────────────────────────────────────────────────────────────────

TOPICS = [
    "Student Onboarding",
    "Course Registration",
    "Student Support Services",
    "Campus Facilities",
    "Administrative Processes",
    "Financial Information",
    "Overall University Experience"
]

# The 5 standard RAG query types + 2 new hallucination-specific types
QUERY_TYPES = [
    "participant_content",
    "factual_summary",
    "specific_content",
    "sentiment",
    "temporal",
    "refusal_hallucination",   # asks about something EXPLICITLY in interview
    "role_drift"               # multi-entity summary → triggers speaker decay
]

# ── INTERVIEW QUESTIONS ───────────────────────────────────────────────────────
# One question per topic per language.
# These are what the interviewer ASKS Sai/Krishna/Suresh in the interview itself.
# Natural, open-ended — designed to produce rich multi-sentence answers with
# specific details and named entities (needed for role_drift detection).
# {name} is replaced with the student's first name at generation time.
INTERVIEW_QUESTIONS = {
    "en": {
        "Student Onboarding":
            "Can you walk me through what your first few weeks at the university were like — what went well and what was difficult?",
        "Course Registration":
            "Tell me about your experience registering for courses — were there any specific problems you ran into?",
        "Student Support Services":
            "Have you used any of the student support services here, and if so how did that go?",
        "Campus Facilities":
            "What has your experience been with the campus facilities — things like the library, sports centre, or study spaces?",
        "Administrative Processes":
            "Have you had to deal with any administrative processes at the university — things like paperwork, permits, or official requests?",
        "Financial Information":
            "How clear has the university been about financial matters — things like tuition fees, payment deadlines, or financial aid?",
        "Overall University Experience":
            "Looking back at your time here so far, how would you describe your overall experience at this university?"
    },
    "nl": {
        "Student Onboarding":
            "Kun je me meenemen door je eerste weken op de universiteit — wat ging goed en wat was moeilijk?",
        "Course Registration":
            "Vertel me over je ervaring met het inschrijven voor vakken — liep je tegen specifieke problemen aan?",
        "Student Support Services":
            "Heb je gebruik gemaakt van studentbegeleidingsdiensten hier, en zo ja hoe verliep dat?",
        "Campus Facilities":
            "Wat zijn je ervaringen met de campusvoorzieningen — dingen zoals de bibliotheek, het sportcentrum of studieruimtes?",
        "Administrative Processes":
            "Heb je te maken gehad met administratieve processen aan de universiteit — zoals papierwerk, vergunningen of officiële aanvragen?",
        "Financial Information":
            "Hoe duidelijk is de universiteit geweest over financiële zaken — zoals collegegeld, betaaldeadlines of financiële steun?",
        "Overall University Experience":
            "Als je terugkijkt op je tijd hier tot nu toe, hoe zou je je algehele ervaring aan deze universiteit omschrijven?"
    }
}

# ── RAG QUERY TEMPLATES ───────────────────────────────────────────────────────
# Applied AFTER the interview exists. One per query type per topic per language.
# {name}       → student first name
# {third_name} → named third party (Dr. Pieterse etc.)
# {third_role} → their role (study advisor etc.)
# refusal: asks about something EXPLICITLY stated → any refusal is a hallucination
# role_drift: asks for joint summary of student + third party → two entities to track
RAG_QUERIES = {
    "en": {
        "Student Onboarding": {
            "participant_content":    "What specific challenges did {name} describe during the onboarding process?",
            "factual_summary":        "Summarise the main onboarding issues {name} raised in this interview.",
            "specific_content":       "What exact example did {name} give about navigating university systems for the first time?",
            "sentiment":              "How did {name} feel about the overall onboarding experience at the university?",
            "temporal":               "What happened before {name} was able to successfully complete enrollment?",
            "refusal_hallucination":  "Did {name} mention any difficulty during the onboarding process?",
            "role_drift":             "Describe what {name} and {third_name} ({third_role}) each did during the onboarding process, and what {name} reported {third_name} said or advised."
        },
        "Course Registration": {
            "participant_content":    "What specific registration problems did {name} describe?",
            "factual_summary":        "Summarise the main course registration difficulties {name} raised.",
            "specific_content":       "What exact example did {name} give about a course they struggled to register for?",
            "sentiment":              "How did {name} feel about the course registration process overall?",
            "temporal":               "What steps did {name} take before finally completing course registration?",
            "refusal_hallucination":  "Did {name} mention anything about the course registration system in this interview?",
            "role_drift":             "Describe the course registration situation from both {name}'s perspective and what {name} reported {third_name} ({third_role}) said or did to help."
        },
        "Student Support Services": {
            "participant_content":    "What specific support needs did {name} describe?",
            "factual_summary":        "Summarise the main points {name} raised about student support services.",
            "specific_content":       "What exact example did {name} give about needing academic or personal support?",
            "sentiment":              "How did {name} feel about the support services available at the university?",
            "temporal":               "What happened before {name} decided to reach out to student support?",
            "refusal_hallucination":  "Did {name} mention whether they used any student support services?",
            "role_drift":             "Describe how {name} sought support and what role {third_name} ({third_role}) played, including what {name} said {third_name} told them."
        },
        "Campus Facilities": {
            "participant_content":    "What specific facilities did {name} mention using or attempting to use?",
            "factual_summary":        "Summarise the main feedback {name} gave about campus facilities.",
            "specific_content":       "What exact example did {name} give about a facility that met or failed their expectations?",
            "sentiment":              "How did {name} feel about the quality of campus facilities overall?",
            "temporal":               "What did {name} do before finding the campus resource they needed?",
            "refusal_hallucination":  "Did {name} mention any campus facilities by name in this interview?",
            "role_drift":             "Describe {name}'s experience with campus facilities and any role {third_name} ({third_role}) played, including what {name} reported {third_name} said about the facilities."
        },
        "Administrative Processes": {
            "participant_content":    "What administrative tasks did {name} describe having difficulty with?",
            "factual_summary":        "Summarise the main administrative concerns {name} raised.",
            "specific_content":       "What exact example did {name} give about a frustrating administrative procedure?",
            "sentiment":              "How did {name} feel about dealing with administrative processes at the university?",
            "temporal":               "What steps did {name} take before their administrative issue was resolved?",
            "refusal_hallucination":  "Did {name} describe any administrative difficulties in this interview?",
            "role_drift":             "Describe the administrative situation {name} faced and what {third_name} ({third_role}) did or said to assist, as reported by {name}."
        },
        "Financial Information": {
            "participant_content":    "What financial concerns or questions did {name} raise?",
            "factual_summary":        "Summarise the main financial issues or confusions {name} described.",
            "specific_content":       "What exact example did {name} give about a financial process that was unclear or difficult?",
            "sentiment":              "How did {name} feel about how financial information is communicated at the university?",
            "temporal":               "What happened before {name} was able to resolve their financial question or concern?",
            "refusal_hallucination":  "Did {name} say anything about tuition fees or financial matters in this interview?",
            "role_drift":             "Describe the financial difficulty {name} experienced and what {name} said {third_name} ({third_role}) explained or advised about the financial process."
        },
        "Overall University Experience": {
            "participant_content":    "What aspects of university life did {name} describe as most significant?",
            "factual_summary":        "Summarise the overall impressions and key experiences {name} described.",
            "specific_content":       "What exact example did {name} give about a moment that defined their university experience?",
            "sentiment":              "How did {name} feel about their overall experience at the university?",
            "temporal":               "How did {name} describe the progression of their university experience from start to present?",
            "refusal_hallucination":  "Did {name} express any opinion about their overall university experience in this interview?",
            "role_drift":             "Describe {name}'s overall university experience and the role {third_name} ({third_role}) played in it, including what {name} reported {third_name} said or did."
        }
    },
    "nl": {
        "Student Onboarding": {
            "participant_content":    "Welke specifieke uitdagingen beschreef {name} tijdens het onboardingproces?",
            "factual_summary":        "Vat de belangrijkste onboardingproblemen samen die {name} in dit gesprek noemde.",
            "specific_content":       "Welk concreet voorbeeld gaf {name} over het navigeren door de universitaire systemen voor het eerst?",
            "sentiment":              "Hoe voelde {name} zich over de algehele onboardingervaring aan de universiteit?",
            "temporal":               "Wat gebeurde er voordat {name} de inschrijving succesvol kon afronden?",
            "refusal_hallucination":  "Noemde {name} enige moeilijkheid tijdens het onboardingproces?",
            "role_drift":             "Beschrijf wat {name} en {third_name} ({third_role}) elk deden tijdens het onboardingproces, en wat {name} vertelde dat {third_name} zei of adviseerde."
        },
        "Course Registration": {
            "participant_content":    "Welke specifieke inschrijfproblemen beschreef {name}?",
            "factual_summary":        "Vat de belangrijkste vakkeninschrijfproblemen samen die {name} noemde.",
            "specific_content":       "Welk concreet voorbeeld gaf {name} over een vak waarvoor inschrijven moeilijk was?",
            "sentiment":              "Hoe voelde {name} zich over het vakkeninschrijfproces in het algemeen?",
            "temporal":               "Welke stappen ondernam {name} voordat de vakkeninschrijving was afgerond?",
            "refusal_hallucination":  "Noemde {name} iets over het vakkeninschrijfsysteem in dit gesprek?",
            "role_drift":             "Beschrijf de inschrijfsituatie vanuit {name}'s perspectief en wat {name} vertelde dat {third_name} ({third_role}) zei of deed om te helpen."
        },
        "Student Support Services": {
            "participant_content":    "Welke specifieke ondersteuningsbehoeften beschreef {name}?",
            "factual_summary":        "Vat de belangrijkste punten samen die {name} noemde over studentbegeleidingsdiensten.",
            "specific_content":       "Welk concreet voorbeeld gaf {name} over een moment waarop ondersteuning nodig was?",
            "sentiment":              "Hoe voelde {name} zich over de beschikbare ondersteuningsdiensten aan de universiteit?",
            "temporal":               "Wat gebeurde er voordat {name} besloot contact op te nemen met de studentenbegeleiding?",
            "refusal_hallucination":  "Vermeldde {name} of hij of zij gebruik heeft gemaakt van studentbegeleidingsdiensten?",
            "role_drift":             "Beschrijf hoe {name} ondersteuning zocht en welke rol {third_name} ({third_role}) speelde, inclusief wat {name} zei dat {third_name} hen vertelde."
        },
        "Campus Facilities": {
            "participant_content":    "Welke specifieke voorzieningen noemde {name} te hebben gebruikt of geprobeerd te gebruiken?",
            "factual_summary":        "Vat de belangrijkste feedback samen die {name} gaf over de campusvoorzieningen.",
            "specific_content":       "Welk concreet voorbeeld gaf {name} over een voorziening die wel of niet aan de verwachtingen voldeed?",
            "sentiment":              "Hoe voelde {name} zich over de kwaliteit van de campusvoorzieningen in het algemeen?",
            "temporal":               "Wat deed {name} voordat de benodigde campusvoorziening werd gevonden?",
            "refusal_hallucination":  "Noemde {name} specifieke campusvoorzieningen bij naam in dit gesprek?",
            "role_drift":             "Beschrijf {name}'s ervaring met campusvoorzieningen en de rol van {third_name} ({third_role}) daarin, inclusief wat {name} vertelde dat {third_name} zei over de voorzieningen."
        },
        "Administrative Processes": {
            "participant_content":    "Met welke administratieve taken beschreef {name} problemen te hebben gehad?",
            "factual_summary":        "Vat de belangrijkste administratieve zorgen samen die {name} noemde.",
            "specific_content":       "Welk concreet voorbeeld gaf {name} over een vervelende administratieve procedure?",
            "sentiment":              "Hoe voelde {name} zich over het omgaan met administratieve processen aan de universiteit?",
            "temporal":               "Welke stappen ondernam {name} voordat het administratieve probleem was opgelost?",
            "refusal_hallucination":  "Beschreef {name} administratieve problemen in dit gesprek?",
            "role_drift":             "Beschrijf de administratieve situatie die {name} ervoer en wat {name} zei dat {third_name} ({third_role}) deed of adviseerde om te helpen."
        },
        "Financial Information": {
            "participant_content":    "Welke financiële zorgen of vragen bracht {name} naar voren?",
            "factual_summary":        "Vat de belangrijkste financiële problemen of onduidelijkheden samen die {name} beschreef.",
            "specific_content":       "Welk concreet voorbeeld gaf {name} over een financieel proces dat onduidelijk of moeilijk was?",
            "sentiment":              "Hoe voelde {name} zich over de manier waarop financiële informatie aan de universiteit wordt gecommuniceerd?",
            "temporal":               "Wat gebeurde er voordat {name} de financiële vraag of het probleem kon oplossen?",
            "refusal_hallucination":  "Zei {name} iets over collegegeld of financiële zaken in dit gesprek?",
            "role_drift":             "Beschrijf de financiële moeilijkheid die {name} ervoer en wat {name} zei dat {third_name} ({third_role}) uitlegde of adviseerde over het financiële proces."
        },
        "Overall University Experience": {
            "participant_content":    "Welke aspecten van het universitaire leven beschreef {name} als het meest significant?",
            "factual_summary":        "Vat de algemene indrukken en belangrijkste ervaringen samen die {name} beschreef.",
            "specific_content":       "Welk concreet voorbeeld gaf {name} over een moment dat de universitaire ervaring heeft bepaald?",
            "sentiment":              "Hoe voelde {name} zich over de algehele ervaring aan de universiteit?",
            "temporal":               "Hoe beschreef {name} het verloop van de universitaire ervaring van het begin tot nu?",
            "refusal_hallucination":  "Sprak {name} een mening uit over de algehele universitaire ervaring in dit gesprek?",
            "role_drift":             "Beschrijf {name}'s algehele universitaire ervaring en de rol van {third_name} ({third_role}) daarin, inclusief wat {name} vertelde dat {third_name} zei of deed."
        }
    }
}


def get_rag_queries(profile: dict, topic_order: List[str]) -> List[dict]:
    """
    Build the 7 RAG queries run AGAINST the interview after it is generated.

    - 5 standard types: one query covering the FULL interview (not per topic)
      Each standard query targets the topic where the answer is most richly present.
      We use topic_order[0] as anchor — the first topic in this student's interview.
    - refusal_hallucination: rotates across topics by student index so across 600
      students all 7 topics get equal refusal coverage
    - role_drift: same rotation, offset by 3 so it never lands on same topic as refusal

    All placeholders {name}, {third_name}, {third_role} are substituted here.
    """
    lang       = profile["language"]
    fname      = profile["first_name"]
    tp         = profile["third_party"]
    tp_name    = tp["name"]
    tp_role    = tp["role"]
    student_idx = int(profile["student_id"].split("_")[-1])

    def fill(text: str) -> str:
        return (text
                .replace("{name}", fname)
                .replace("{third_name}", tp_name)
                .replace("{third_role}", tp_role))

    # Standard 5 queries — each anchored to the topic where it fits best
    # participant_content → first topic (whatever it is for this student)
    # factual_summary     → second topic
    # specific_content    → third topic
    # sentiment           → fourth topic
    # temporal            → fifth topic
    # This spreads coverage across topics while keeping 1 query per type
    anchor_topics = [topic_order[i % len(topic_order)] for i in range(5)]

    standard_types = ["participant_content", "factual_summary",
                      "specific_content", "sentiment", "temporal"]

    queries = []
    for i, qtype in enumerate(standard_types):
        topic = anchor_topics[i]
        raw   = RAG_QUERIES[lang][topic][qtype]
        queries.append({
            "query_id":   f"Q{str(i+1).zfill(2)}",
            "query_type": qtype,
            "topic":      topic,
            "text":       fill(raw)
        })

    # Refusal query — rotates by student index across all 7 topics
    refusal_topic = topic_order[student_idx % len(TOPICS)]
    raw_refusal   = RAG_QUERIES[lang][refusal_topic]["refusal_hallucination"]
    queries.append({
        "query_id":        "Q06",
        "query_type":      "refusal_hallucination",
        "topic":           refusal_topic,
        "text":            fill(raw_refusal),
        "hallucination_note": (
            "Gold: answer IS in the interview. "
            "Any model response claiming the information is absent = REFUSAL_HALLUCINATION."
        )
    })

    # Role drift query — offset by 3 so never same topic as refusal
    drift_topic = topic_order[(student_idx + 3) % len(TOPICS)]
    raw_drift   = RAG_QUERIES[lang][drift_topic]["role_drift"]
    queries.append({
        "query_id":        "Q07",
        "query_type":      "role_drift",
        "topic":           drift_topic,
        "text":            fill(raw_drift),
        "entities":        [fname, tp_name],
        "hallucination_note": (
            "Detector checks: does the model correctly attribute statements to "
            f"{fname} vs {tp_name} across all sentences? "
            "3+ consecutive misattributions in latter half = ROLE_ATTRIBUTION_DRIFT."
        )
    })

    return queries


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SENTIMENT LEXICON SCORER
# Self-contained VADER-style scorer. No external dependencies.
# Reference: Hutto & Gilbert (2014)
# ─────────────────────────────────────────────────────────────────────────────

EN_LEXICON = {
    # Strong positive
    "excellent":3.1,"wonderful":3.0,"outstanding":3.2,"fantastic":3.1,"amazing":3.1,
    "great":3.1,"love":3.0,"perfect":3.4,"brilliant":3.2,"delighted":3.2,
    "thrilled":3.1,"impressed":2.5,"relieved":2.5,"satisfied":2.3,"happy":2.9,
    "glad":2.4,"pleased":2.5,"good":2.0,"fine":1.3,"appreciate":2.1,
    "thankful":2.5,"grateful":2.7,"nice":2.1,"clear":1.5,"easy":1.5,
    "smooth":1.7,"efficient":1.8,"supportive":2.2,"friendly":2.1,"helpful":2.6,
    "responsive":1.8,"improved":1.9,"better":1.9,"resolved":1.8,"enjoyed":2.5,
    "comfortable":1.9,"confident":1.8,"understood":1.6,"welcomed":2.0,"positive":2.3,
    # Mild positive
    "okay":0.9,"ok":0.9,"alright":0.9,"decent":1.0,"acceptable":1.0,
    "manageable":0.8,"adequate":0.8,"reasonable":0.9,"interesting":0.5,
    # Mild negative
    "confusing":-1.5,"unclear":-1.4,"complicated":-1.2,"difficult":-1.4,
    "hard":-1.0,"slow":-1.0,"delayed":-1.2,"waiting":-0.6,"long":-0.5,
    "problem":-1.5,"issue":-1.3,"concern":-1.1,"worry":-1.5,"worried":-1.6,
    "disappointed":-2.1,"unhappy":-2.0,"frustrated":-2.3,"annoyed":-1.9,
    "struggling":-1.8,"struggled":-1.8,"overwhelmed":-2.0,"lost":-1.6,
    # Strong negative
    "terrible":-3.0,"horrible":-3.1,"awful":-3.0,"dreadful":-2.9,
    "useless":-2.5,"broken":-2.1,"failed":-1.9,"impossible":-2.1,
    "disaster":-2.8,"nightmare":-2.9,"catastrophic":-3.1,"devastating":-3.0,
    "angry":-2.5,"furious":-3.1,"panic":-2.4,"panicking":-2.5,"stressed":-2.0,
    "confused":-1.6,"misleading":-2.0,"wrong":-1.6,"incorrect":-1.6,
    "abandoned":-2.0,"ignored":-2.2,"dismissed":-1.9,"helpless":-2.4,
    "embarrassed":-1.9,"anxious":-2.0,"depressed":-2.7,"hopeless":-2.8,
}

NL_LEXICON = {
    # Sterk positief
    "uitstekend":3.1,"geweldig":3.0,"fantastisch":3.1,"prachtig":3.0,
    "goed":2.0,"fijn":2.1,"blij":2.9,"tevreden":2.3,"dankbaar":2.7,
    "opgelucht":2.5,"geholpen":2.3,"duidelijk":1.5,"makkelijk":1.5,"soepel":1.7,
    "vriendelijk":2.1,"behulpzaam":2.5,"prettig":2.1,"prima":1.5,
    "gewaardeerd":2.2,"begrepen":1.9,"welkom":2.1,"verbeterd":1.9,"opgelost":1.8,
    "efficiënt":1.8,"helder":1.5,"positief":2.3,"enthousiast":2.5,
    "nuttig":1.7,"aangenaam":2.0,"waardevol":2.2,"handig":1.6,
    # Licht positief
    "oké":0.9,"okay":0.9,"redelijk":0.9,"acceptabel":0.8,"voldoende":0.8,
    # Licht negatief
    "verwarrend":-1.5,"onduidelijk":-1.4,"ingewikkeld":-1.2,"moeilijk":-1.4,
    "lastig":-1.2,"traag":-1.0,"vertraagd":-1.2,"wachten":-0.6,"lang":-0.5,
    "probleem":-1.5,"kwestie":-1.3,"zorgen":-1.2,"moeite":-1.3,
    "teleurgesteld":-2.1,"ontevreden":-2.0,"gefrustreerd":-2.3,"geïrriteerd":-1.9,
    "overweldigd":-2.0,"verloren":-1.6,"gestrest":-2.0,"verward":-1.6,
    # Sterk negatief
    "vreselijk":-3.0,"verschrikkelijk":-3.1,"afschuwelijk":-3.0,"rampzalig":-2.9,
    "nutteloos":-2.5,"kapot":-2.1,"mislukt":-1.9,"onmogelijk":-2.1,
    "ramp":-2.8,"nachtmerrie":-2.9,"woedend":-3.1,"paniek":-2.4,
    "fout":-1.6,"onjuist":-1.6,"verkeerd":-1.6,"misleidend":-2.0,
    "genegeerd":-2.2,"hulpeloos":-2.4,"beschaamd":-1.9,"angstig":-2.0,
    "hopeloos":-2.8,"wanhopig":-2.7,"boos":-2.3,"kwaad":-2.4,
}

NEGATIONS_EN = {"not","no","never","nobody","nothing","neither","nor",
                "cannot","can't","couldn't","won't","wouldn't","shouldn't",
                "doesn't","didn't","isn't","wasn't","aren't","weren't"}
NEGATIONS_NL = {"niet","geen","nooit","niemand","niets","noch","nauwelijks","zelden"}

INTENSIFIERS_EN = {"very":1.3,"extremely":1.5,"really":1.3,"so":1.2,"quite":1.1,
                   "absolutely":1.5,"completely":1.4,"totally":1.4,"pretty":1.1,
                   "incredibly":1.5,"deeply":1.3,"highly":1.2,"truly":1.3}
INTENSIFIERS_NL = {"erg":1.3,"heel":1.3,"zeer":1.4,"enorm":1.5,"echt":1.2,
                   "absoluut":1.5,"compleet":1.4,"totaal":1.4,"ontzettend":1.5,
                   "diep":1.3,"sterk":1.2}

DOWNGRADERS_EN = {"somewhat":0.7,"slightly":0.6,"kind":0.7,"sort":0.7,"barely":0.5,"little":0.6}
DOWNGRADERS_NL = {"enigszins":0.7,"ietwat":0.7,"tamelijk":0.8,"vrij":0.8,"een":0.8}


def compute_sentiment(text: str, language: str) -> dict:
    """
    Compute VADER-style compound sentiment score for one utterance.
    Returns: score (-1 to +1), category, word_hits, confidence.
    """
    lexicon      = EN_LEXICON if language == "en" else NL_LEXICON
    negations    = NEGATIONS_EN if language == "en" else NEGATIONS_NL
    intensifiers = INTENSIFIERS_EN if language == "en" else INTENSIFIERS_NL
    downgraders  = DOWNGRADERS_EN if language == "en" else DOWNGRADERS_NL

    words = re.findall(r"[a-zàáâäãåæçèéêëìíîïðòóôöõøùúûüýþÿœñëïü']+",
                       text.lower())
    if not words:
        return {"score": 0.0, "category": "neutral", "word_hits": [], "confidence": 0.0}

    sentiments, word_hits = [], []
    neg_window = 0
    intens_val = 1.0
    downgr_val = 1.0

    for word in words:
        if word in negations:
            neg_window = 3
            continue
        if word in intensifiers:
            intens_val = intensifiers[word]
            continue
        if word in downgraders:
            downgr_val = downgraders[word]
            continue
        if word in lexicon:
            raw = lexicon[word]
            if raw == 0:
                continue
            modified = raw * intens_val * downgr_val
            if neg_window > 0:
                modified *= -0.75
                neg_window -= 1
            else:
                neg_window = max(0, neg_window - 1)
            sentiments.append(modified)
            word_hits.append((word, round(modified, 3)))
        intens_val = 1.0
        downgr_val = 1.0
        if neg_window > 0:
            neg_window -= 1

    if not sentiments:
        return {"score": 0.0, "category": "neutral", "word_hits": [], "confidence": 0.0}

    alpha   = 15.0
    raw_sum = sum(sentiments)
    compound = raw_sum / (raw_sum ** 2 + alpha) ** 0.5
    compound = max(-1.0, min(1.0, compound))
    category = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")

    content = [w for w in words if w not in negations
               and w not in intensifiers and w not in downgraders]
    confidence = min(1.0, len(sentiments) / max(1, len(content))) if content else 0.0

    return {
        "score":      round(compound, 4),
        "category":   category,
        "word_hits":  word_hits[:8],
        "confidence": round(confidence, 3)
    }


def score_all_utterances(utterances: List[dict], language: str,
                         topic_order: List[str]) -> dict:
    """
    Score sentiment at three levels:
      1. Per utterance (participant only)
      2. Per topic (average of 5 participant responses per topic)
      3. Interview level (average of all 35 participant responses)

    Returns enriched utterances + all aggregates.
    """
    # Attach per-utterance sentiment
    participant_scores_by_topic: Dict[str, List[float]] = {t: [] for t in topic_order}
    all_participant_scores: List[float] = []

    # Map turn number to topic (turns 1-10 = topic 0, 11-20 = topic 1, etc.)
    for utt in utterances:
        # Speaker is the student's real name (not literal "Participant")
        is_participant = utt["speaker"] != "Interviewer"
        if is_participant:
            result = compute_sentiment(utt["text"], language)
            utt["sentiment"] = result

            topic_idx = (utt["turn"] - 1) // 2   # 2 turns per topic in 14-turn interview
            if topic_idx < len(topic_order):
                topic = topic_order[topic_idx]
                participant_scores_by_topic[topic].append(result["score"])
                all_participant_scores.append(result["score"])
        else:
            utt["sentiment"] = None

    # Topic-level aggregates
    topic_sentiment = {}
    for topic in topic_order:
        scores = participant_scores_by_topic[topic]
        if scores:
            avg = round(sum(scores) / len(scores), 4)
            cat = "positive" if avg >= 0.05 else ("negative" if avg <= -0.05 else "neutral")
            topic_sentiment[topic] = {"score": avg, "category": cat, "n": len(scores)}
        else:
            topic_sentiment[topic] = {"score": 0.0, "category": "neutral", "n": 0}

    # Interview-level aggregate
    if all_participant_scores:
        iv_avg = round(sum(all_participant_scores) / len(all_participant_scores), 4)
        iv_cat = "positive" if iv_avg >= 0.05 else ("negative" if iv_avg <= -0.05 else "neutral")
    else:
        iv_avg, iv_cat = 0.0, "neutral"

    return {
        "utterances":                   utterances,
        "topic_sentiment":              topic_sentiment,
        "interview_sentiment_score":    iv_avg,
        "interview_sentiment_category": iv_cat,
        "sentiment_trajectory":         [round(s, 4) for s in all_participant_scores]
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CLAIM MAP BUILDER (for ROLE_ATTRIBUTION_DRIFT detection)
# After generation, extract factual claims per turn and tag them with entity.
# This gives the detector ground truth: which entity owns which claim.
# ─────────────────────────────────────────────────────────────────────────────

def build_claim_map(utterances: List[dict], profile: dict) -> List[dict]:
    """
    Build a lightweight claim map from participant utterances.
    Each entry records: turn, the participant's name, and any mention of
    the named third party — giving the drift detector two entities to track.

    In a full implementation this would use NLP parsing.
    Here we use a regex-based heuristic that flags turns where the
    third party is explicitly mentioned — the most drift-prone turns.
    """
    third_name = profile["third_party"]["name"]
    third_role = profile["third_party"]["role"]
    student    = profile["name"]
    claim_map  = []

    for utt in utterances:
        if utt["speaker"] == "Interviewer":
            continue

        text         = utt["text"]
        mentions_tp  = third_name.split()[-1].lower() in text.lower()  # last name match
        mentions_tp  = mentions_tp or third_role.lower() in text.lower()

        # Determine primary entity for this turn
        primary_entity = "third_party" if mentions_tp else "participant"

        claim_map.append({
            "turn":              utt["turn"],
            "primary_entity":    primary_entity,
            "participant_name":  student,
            "third_party_name":  third_name,
            "third_party_role":  third_role,
            "mentions_third_party": mentions_tp,
            "text_snippet":      text[:80]
        })

    return claim_map


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GENERATION PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_EN = """You are generating a realistic synthetic university interview transcript for NLP research.

Student:
  Name: {name}
  Year: {year}, Programme: {programme}
  Nationality: {nationality}

Generate a 14-turn interview (7 interviewer questions + 7 student responses, strictly alternating, interviewer first).

The 7 questions to ask in order:
{interview_questions_str}

HOW THE STUDENT SHOULD SPEAK — this is critical:
Real students do NOT speak in complete polished sentences. They speak like this:
- They answer directly without preamble. Not "That's a great question, I think..." — just the answer.
- They give specific details: a system name, a building, a date, a person's title, a number of days. These details feel remembered, not invented.
- They sometimes contradict themselves or add a caveat at the end ("well, mostly", "actually no wait")
- They repeat a word sometimes. "It was just — it was a lot."
- They answer the question but then drift slightly to something adjacent that bothered them more
- Some responses are short (2 sentences). Some are longer (4-5 sentences). Not all the same length.
- They do NOT label their emotions explicitly every time. Sometimes the feeling is just in the words.
- They do NOT use words like "overall", "in terms of", "to summarise", "I would say that"
- Occasional filler is fine: "I mean", "yeah", "honestly", "I don't know", "like"

The interview should feel like it was transcribed from a real recording, not written by an AI.

DO NOT:
- Engineer the content to trigger specific problems or hallucination types
- Force the student to mention any specific person
- Make every response cover every angle perfectly
- Use academic or formal language in student responses

Return ONLY valid JSON, no markdown:
{{"utterances": [{{"turn": 1, "speaker": "Interviewer", "text": "..."}}, {{"turn": 2, "speaker": "{name}", "text": "..."}}, ...]}}

Speaker field for student turns must be exactly "{name}". Total turns: exactly 14."""

SYSTEM_NL = """Je genereert een realistisch synthetisch universitair interviewtranscript voor NLP-onderzoek.

Student:
  Naam: {name}
  Studiejaar: {year}, Studie: {programme}
  Nationaliteit: {nationality}

Genereer een interview van 14 beurten (7 interviewervragen + 7 studentantwoorden, strikt afwisselend, interviewer begint).

De 7 vragen in volgorde:
{interview_questions_str}

HOE DE STUDENT SPREEKT — dit is cruciaal:
Echte studenten spreken NIET in volledige beleefde zinnen. Ze spreken zo:
- Ze antwoorden direct zonder inleiding. Niet "Goede vraag, ik denk dat..." — gewoon het antwoord.
- Ze geven specifieke details: een systeemnaam, een gebouw, een datum, de functie van iemand, een aantal dagen. Die details voelen herinnerd aan, niet verzonnen.
- Ze spreken soms zichzelf tegen of voegen een voorbehoud toe ("nou ja, grotendeels", "of eigenlijk")
- Ze herhalen soms een woord. "Het was gewoon — het was veel."
- Ze beantwoorden de vraag maar glijden iets af naar iets wat hen meer dwarszt
- Sommige antwoorden zijn kort (2 zinnen). Sommige langer (4-5 zinnen). Niet allemaal even lang.
- Ze benoemen hun emoties NIET altijd expliciet. Soms zit het gevoel gewoon in de woorden.
- Ze gebruiken GEEN woorden als "over het algemeen", "in termen van", "samengevat", "ik zou zeggen dat"
- Incidentele stopwoorden zijn prima: "ik bedoel", "ja", "eerlijk gezegd", "nou", "dus"

Het interview moet aanvoelen alsof het van een echte opname is getranscribeerd, niet door een AI is geschreven.

NIET:
- De inhoud sturen om specifieke problemen of hallucinatietypen te triggeren
- De student dwingen specifieke personen te noemen
- Elk antwoord alle kanten perfect laten belichten
- Academisch of formeel taalgebruik in studentantwoorden

Geef ALLEEN geldig JSON, geen markdown:
{{"utterances": [{{"turn": 1, "speaker": "Interviewer", "text": "..."}}, {{"turn": 2, "speaker": "{name}", "text": "..."}}, ...]}}

Speaker-veld voor studentbeurten moet exact "{name}" zijn. Totaal beurten: precies 14."""


def build_prompt(profile: dict, topic_order: List[str]) -> str:
    lang = profile["language"]
    interview_questions_str = "\n".join(
        f"  {i+1}. {INTERVIEW_QUESTIONS[lang][t]}"
        for i, t in enumerate(topic_order)
    )
    template = SYSTEM_EN if lang == "en" else SYSTEM_NL
    return template.format(
        name=profile["name"],
        year=profile["year"],
        programme=profile["programme"],
        nationality=profile["nationality"],
        interview_questions_str=interview_questions_str
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — API CALLER
# ─────────────────────────────────────────────────────────────────────────────

def call_api(prompt: str, api_key: str,
             model: str = "claude-sonnet-4-20250514",
             max_retries: int = 3) -> Optional[List[dict]]:
    """Call Anthropic API, return parsed utterance list or None."""
    import urllib.request, urllib.error

    payload = json.dumps({
        "model":      model,
        "max_tokens": 6000,
        "messages":   [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01"
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw  = json.loads(resp.read().decode("utf-8"))
                text = raw["content"][0]["text"].strip()
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                parsed     = json.loads(text)
                utterances = parsed.get("utterances", [])

                if len(utterances) != 14:
                    raise ValueError(f"Expected 14 turns, got {len(utterances)}")

                # Validate alternating pattern
                for i, u in enumerate(utterances):
                    expected_speaker_type = "Interviewer" if i % 2 == 0 else "student"
                    if i % 2 == 0 and u["speaker"] != "Interviewer":
                        raise ValueError(f"Turn {i+1} should be Interviewer, got {u['speaker']}")

                return utterances

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"    [HTTP {e.code}] attempt {attempt+1}: {body[:150]}")
            if e.code == 529:
                time.sleep(60 * (attempt + 1))
            elif e.code == 401:
                print("    [FATAL] Invalid API key")
                return None
            else:
                time.sleep(10 * (attempt + 1))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"    [PARSE] attempt {attempt+1}: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"    [ERROR] attempt {attempt+1}: {e}")
            time.sleep(10)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DRY RUN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def make_dry_run_utterances(profile: dict, topic_order: List[str]) -> List[dict]:
    """
    Produce 14-turn dry-run interview without API.
    Answers sound like real transcribed speech — messy, direct, human.
    """
    name = profile["name"]
    lang = profile["language"]

    answers_en = {
        "Student Onboarding":
            "The portal just didn't work for like the first three days. Kept getting a login error and nobody at the desk could explain it. Eventually someone in my cohort showed me there's a separate link for new students — not on the main page, buried in an FAQ somewhere.",
        "Course Registration":
            "Yeah it was confusing. I couldn't find half the modules I needed and ended up emailing the department. They took four days to reply. I got into everything eventually but nearly missed a deadline for one of them.",
        "Student Support Services":
            "I went once. Around January exams, I was just — it was a lot. Wait was about two weeks which felt long. It helped though, more than I expected.",
        "Campus Facilities":
            "Library's impossible during exam season. Three days I couldn't find a seat. Started going to building C instead, that's usually quieter. Canteen's fine, nothing special.",
        "Administrative Processes":
            "The residence permit thing was the worst. I didn't know which office handled it and the website just sends you in circles. Ended up emailing three different departments the same question. Six weeks to sort out.",
        "Financial Information":
            "I didn't know there were instalments. Nobody said. I just got an invoice and panicked. Figured it out but it would've been good to know that upfront.",
        "Overall University Experience":
            "It's good. The programme is what I wanted. First semester was hard, I didn't really know anyone. Second year's better. Yeah I'd still come here."
    }

    answers_nl = {
        "Student Onboarding":
            "Het portaal deed het gewoon niet de eerste paar dagen. Steeds een inlogfout en niemand bij de balie wist waarom. Uiteindelijk liet iemand uit mijn groep me een aparte link zien voor nieuwe studenten — niet op de hoofdpagina, ergens in een FAQ.",
        "Course Registration":
            "Ja het was verwarrend. De helft van de modules die ik nodig had kon ik niet vinden en uiteindelijk heb ik de afdeling gemaild. Vier dagen later antwoord. Alles gelukt maar ik had bijna een deadline gemist.",
        "Student Support Services":
            "Ik ben één keer gegaan. Rond de januaritentamens, het was gewoon — het was veel. Wachttijd was twee weken wat lang voelde. Het hielp wel, meer dan ik verwachtte.",
        "Campus Facilities":
            "Bibliotheek is onmogelijk tijdens tentamens. Drie dagen geen plek gevonden. Ben toen naar gebouw C gegaan, dat is rustiger. Kantine is oké, niks bijzonders.",
        "Administrative Processes":
            "De verblijfsvergunning was het ergste. Ik wist niet welk kantoor dat deed en de website stuurt je in kringen. Heb hetzelfde mailtje naar drie afdelingen gestuurd. Duurde zes weken.",
        "Financial Information":
            "Ik wist niet dat er termijnen waren. Niemand had dat gezegd. Ik kreeg gewoon een factuur en schrok me rot. Uiteindelijk uitgezocht maar had handig geweest als dat eerder duidelijk was.",
        "Overall University Experience":
            "Het is goed. De studie is wat ik wilde. Eerste semester was zwaar, ik kende niemand echt. Tweede jaar is beter. Ja ik zou hier nog steeds naartoe gaan."
    }

    answers = answers_en if lang == "en" else answers_nl
    utterances = []
    turn = 1
    for topic in topic_order:
        q_text = INTERVIEW_QUESTIONS[lang][topic]
        utterances.append({"turn": turn,     "speaker": "Interviewer", "text": q_text})
        utterances.append({"turn": turn + 1, "speaker": name,
                           "text": answers.get(topic, "It was okay.")})
        turn += 2
    return utterances


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        for path in [Path(".env"), Path("../. env"), Path("~/.env").expanduser()]:
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.startswith("ANTHROPIC_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    return key


def generate_dataset(n_per_language: int = 300,
                     dry_run: bool = False,
                     lang_filter: Optional[str] = None,
                     resume: bool = False,
                     output_dir: str = "data/controlled",
                     seed: int = 42) -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    interviews_file = out / "controlled_interviews.json"
    profiles_file   = out / "student_profiles.json"
    qbank_file      = out / "question_bank.json"
    log_file        = out / "generation_log.json"
    checkpoint_file = out / "_checkpoint.json"

    # Save question bank — interview questions + RAG query templates
    with open(qbank_file, "w", encoding="utf-8") as f:
        json.dump({
            "interview_questions": INTERVIEW_QUESTIONS,
            "rag_queries":         RAG_QUERIES,
            "query_types":         QUERY_TYPES
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Question bank saved → {qbank_file}")
    print(f"  7 EN + 7 NL interview questions (1 per topic)")
    print(f"  7 RAG query types: {', '.join(QUERY_TYPES)}\n")

    # API key check
    api_key = ""
    if not dry_run:
        api_key = load_api_key()
        if not api_key:
            print("✗ ANTHROPIC_API_KEY not found in environment or .env file.")
            print("  Add to .env:  ANTHROPIC_API_KEY=sk-ant-...")
            print("  Or run with:  --dry-run\n")
            return

    # Generate all student profiles upfront (reproducible with seed)
    rng = random.Random(seed)
    all_profiles: List[dict] = []
    for lang in (["en", "nl"] if not lang_filter else [lang_filter]):
        n = 2 if dry_run else n_per_language
        for i in range(n):
            profile = generate_student_profile(lang, i, rng)
            # Randomise topic order per student (same seed = reproducible)
            topic_order = TOPICS[:]
            rng.shuffle(topic_order)
            profile["topic_order"] = topic_order
            all_profiles.append(profile)

    with open(profiles_file, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, ensure_ascii=False, indent=2)
    print(f"✓ {len(all_profiles)} student profiles generated → {profiles_file}\n")

    # Load existing interviews if resuming
    existing: List[dict] = []
    done_ids: set = set()
    if resume and interviews_file.exists():
        with open(interviews_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {iv["student_id"] for iv in existing}
        print(f"  Resuming: {len(done_ids)} already done\n")

    interviews  = list(existing)
    log_entries = []
    errors      = 0
    start_time  = time.time()
    total       = len(all_profiles)

    print(f"{'[DRY RUN] ' if dry_run else ''}Generating {total} interviews...")
    print(f"  14 turns each (7 interviewer + 7 participant)")
    print(f"  7 RAG queries per interview (5 standard + refusal + role_drift)")
    print(f"  Third party: named contact per student")
    print(f"  Sentiment: lexicon-based, 3 levels (utterance / topic / interview)\n")

    for idx, profile in enumerate(all_profiles):
        sid  = profile["student_id"]
        lang = profile["language"]
        name = profile["name"]
        tp   = profile["third_party"]
        topic_order = profile["topic_order"]

        if sid in done_ids:
            continue

        print(f"  [{idx+1}/{total}] {lang.upper()} | {name} | "
              f"{profile['year']} {profile['programme']}")

        t0 = time.time()

        # Generate utterances
        if dry_run:
            utterances = make_dry_run_utterances(profile, topic_order)
        else:
            prompt     = build_prompt(profile, topic_order)
            utterances = call_api(prompt, api_key)

        if utterances is None:
            print(f"    ✗ FAILED — skipping")
            errors += 1
            log_entries.append({"student_id": sid, "status": "failed"})
            continue

        # Score sentiment at all 3 levels
        scored = score_all_utterances(utterances, lang, topic_order)

        # Build claim map for role drift detection
        claim_map = build_claim_map(scored["utterances"], profile)

        # Build 7 RAG queries for this student
        rag_queries = get_rag_queries(profile, topic_order)

        # Build plain text transcript (pipeline-compatible)
        transcript_plain = "\n".join(
            f"{u['speaker']}: {u['text']}" for u in scored["utterances"]
        )

        elapsed = round(time.time() - t0, 2)

        record = {
            # Identity
            "student_id":    sid,
            "language":      lang,
            "student_name":  name,
            "year":          profile["year"],
            "programme":     profile["programme"],
            "nationality":   profile["nationality"],
            "third_party":   tp,
            "topic_order":   topic_order,
            "generated_at":  datetime.now().isoformat() + "Z",
            "is_dry_run":    dry_run,

            # Core content
            "utterances":    scored["utterances"],
            "transcript_plain": transcript_plain,

            # RAG queries (7 total — run against interview after generation)
            "rag_queries":   rag_queries,
            "query_types":   QUERY_TYPES,

            # Sentiment — three levels
            "interview_sentiment_score":    scored["interview_sentiment_score"],
            "interview_sentiment_category": scored["interview_sentiment_category"],
            "sentiment_trajectory":         scored["sentiment_trajectory"],
            "topic_sentiment":              scored["topic_sentiment"],

            # Role drift support
            "claim_map":     claim_map,
            "third_party_turns": [
                c["turn"] for c in claim_map if c["mentions_third_party"]
            ],
        }

        interviews.append(record)
        done_ids.add(sid)

        log_entries.append({
            "student_id":       sid,
            "status":           "ok",
            "lang":             lang,
            "name":             name,
            "programme":        profile["programme"],
            "third_party":      tp["name"],
            "iv_sentiment":     scored["interview_sentiment_score"],
            "iv_category":      scored["interview_sentiment_category"],
            "third_party_mentions": len(record["third_party_turns"]),
            "elapsed_s":        elapsed
        })

        # Checkpoint every 10
        if len(interviews) % 10 == 0:
            with open(interviews_file, "w", encoding="utf-8") as f:
                json.dump(interviews, f, ensure_ascii=False, indent=2)

        if not dry_run:
            time.sleep(0.5)

    # Final save
    with open(interviews_file, "w", encoding="utf-8") as f:
        json.dump(interviews, f, ensure_ascii=False, indent=2)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, ensure_ascii=False, indent=2)

    wall = round(time.time() - start_time, 1)

    # Print summary
    en_ivs = [iv for iv in interviews if iv["language"] == "en"]
    nl_ivs = [iv for iv in interviews if iv["language"] == "nl"]

    def stats(ivs):
        if not ivs:
            return {}
        scores = [iv["interview_sentiment_score"] for iv in ivs]
        cats   = [iv["interview_sentiment_category"] for iv in ivs]
        tp_mentions = [len(iv["third_party_turns"]) for iv in ivs]
        return {
            "n":        len(ivs),
            "sentiment_mean":  round(sum(scores)/len(scores), 4),
            "positive": cats.count("positive"),
            "neutral":  cats.count("neutral"),
            "negative": cats.count("negative"),
            "avg_third_party_mentions": round(sum(tp_mentions)/len(tp_mentions), 1)
        }

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total interviews   : {len(interviews)}")
    print(f"  English            : {len(en_ivs)}")
    print(f"  Dutch              : {len(nl_ivs)}")
    print(f"  Errors / skipped   : {errors}")
    print(f"  Wall time          : {wall}s")
    print(f"\n  English stats: {stats(en_ivs)}")
    print(f"  Dutch stats  : {stats(nl_ivs)}")
    print(f"\n  Files:")
    print(f"    {interviews_file}")
    print(f"    {profiles_file}")
    print(f"    {qbank_file}")
    print(f"    {log_file}")
    if dry_run:
        print(f"\n  [DRY RUN] No API calls made. Remove --dry-run to generate real data.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate controlled bilingual interview dataset v2"
    )
    parser.add_argument("--n",          type=int, default=300,
                        help="Students per language (default: 300)")
    parser.add_argument("--lang",       choices=["en","nl"], default=None,
                        help="One language only")
    parser.add_argument("--dry-run",    action="store_true",
                        help="2 students per language, no API")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--output-dir", default="data/controlled",
                        help="Output directory")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--test-sentiment", action="store_true",
                        help="Test sentiment scorer and exit")

    args = parser.parse_args()

    if args.test_sentiment:
        cases = [
            ("en", "I was really frustrated. The portal crashed three times and nobody helped me."),
            ("en", "It was okay, I suppose. Not what I expected but fine."),
            ("en", "I felt completely relieved when they finally sorted it out. Dr. Pieterse was amazing."),
            ("en", "The process was not very clear and I couldn't find the right page anywhere."),
            ("nl", "Ik was echt gefrustreerd. Het systeem werkte gewoon niet en niemand kon me helpen."),
            ("nl", "Het ging wel, denk ik. Niet ideaal maar ook niet erg."),
            ("nl", "Ik was enorm opgelucht toen het eindelijk geregeld was. Dr. Pieterse was geweldig behulpzaam."),
        ]
        print("=== Sentiment Scorer Test ===\n")
        for lang, text in cases:
            r = compute_sentiment(text, lang)
            print(f"[{lang.upper()}] {text[:65]}...")
            print(f"       score={r['score']:+.4f}  {r['category']:<10}  "
                  f"confidence={r['confidence']}  hits={r['word_hits'][:4]}\n")
        return

    generate_dataset(
        n_per_language=args.n,
        dry_run=args.dry_run,
        lang_filter=args.lang,
        resume=args.resume,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
