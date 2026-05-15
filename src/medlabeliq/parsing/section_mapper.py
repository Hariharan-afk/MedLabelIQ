from __future__ import annotations

import re


# Code-first canonical mapping using official FDA SPL section-heading LOINC codes.
# We intentionally keep this focused on sections that matter for downstream retrieval.
LOINC_TO_CANONICAL_SECTION: dict[str, str] = {
    "34066-1": "boxed_warning",
    "43683-2": "recent_major_changes",
    "34067-9": "indications_and_usage",
    "34068-7": "dosage_and_administration",
    "43678-2": "dosage_forms_and_strengths",
    "34070-3": "contraindications",
    "43685-7": "warnings_and_precautions",
    "34071-1": "warnings",
    "42232-9": "precautions",
    "34084-4": "adverse_reactions",
    "34073-7": "drug_interactions",
    "43684-0": "use_in_specific_populations",
    "42228-7": "pregnancy",
    "34080-2": "nursing_mothers",
    "34081-0": "pediatric_use",
    "34082-8": "geriatric_use",
    "34088-5": "overdosage",
    "34089-3": "description",
    "34090-1": "clinical_pharmacology",
    "43679-0": "mechanism_of_action",
    "43681-6": "pharmacodynamics",
    "43682-4": "pharmacokinetics",
    "43680-8": "nonclinical_toxicology",
    "34083-6": "carcinogenesis_mutagenesis_impairment_of_fertility",
    "34092-7": "clinical_studies",
    "34069-5": "how_supplied_storage_and_handling",
    "34076-0": "patient_counseling_information",
    "42231-1": "medication_guide",
    "42230-3": "patient_package_insert",
    "38056-8": "supplemental_patient_material",
    "48780-1": "spl_product_data_elements",
    "48779-3": "spl_indexing_data_elements",
    "69718-5": "statement_of_identity",
    "44425-7": "storage_and_handling",

    # Additional standardized clinical / label sections found during Step 4C analysis
    "34072-9": "general_precautions",
    "34075-2": "laboratory_tests",
    "34079-4": "labor_and_delivery",
    "34093-5": "references",
    "49489-8": "microbiology",
    "51945-4": "principal_display_panel",
    "69759-9": "risk_summary",
    "88828-9": "renal_impairment",
    "88829-7": "hepatic_impairment",
    "90375-7": "postmarketing_experience",

    # OTC Drug Facts sections
    "55106-9": "active_ingredient",
    "55105-1": "purpose",
    "50565-1": "keep_out_of_reach_of_children",
    "50566-9": "stop_use_and_ask_doctor",
    "50567-7": "when_using_this_product",
    "50568-5": "ask_doctor_or_pharmacist_before_use",
    "50569-3": "ask_doctor_before_use",
    "50570-1": "do_not_use",
    "53413-1": "questions_or_comments",
    "53414-9": "pregnancy_or_breastfeeding_otc",
    "51727-6": "inactive_ingredients",
    "59845-8": "instructions_for_use",
}


# Title fallback is required because:
# 1. some legacy sections use titles more reliably than codes,
# 2. code 42229-5 means "SPL UNCLASSIFIED SECTION",
# 3. many useful subsections are unclassified but still have meaningful titles.
TITLE_TO_CANONICAL_SECTION: dict[str, str] = {
    "BOXED WARNING": "boxed_warning",
    "RECENT MAJOR CHANGES": "recent_major_changes",
    "INDICATIONS AND USAGE": "indications_and_usage",
    "DOSAGE AND ADMINISTRATION": "dosage_and_administration",
    "DOSAGE FORMS AND STRENGTHS": "dosage_forms_and_strengths",
    "CONTRAINDICATIONS": "contraindications",
    "WARNINGS AND PRECAUTIONS": "warnings_and_precautions",
    "WARNINGS": "warnings",
    "PRECAUTIONS": "precautions",
    "ADVERSE REACTIONS": "adverse_reactions",
    "DRUG INTERACTIONS": "drug_interactions",
    "USE IN SPECIFIC POPULATIONS": "use_in_specific_populations",
    "PREGNANCY": "pregnancy",
    "LACTATION": "lactation",
    "FEMALES AND MALES OF REPRODUCTIVE POTENTIAL": "reproductive_potential",
    "PEDIATRIC USE": "pediatric_use",
    "GERIATRIC USE": "geriatric_use",
    "OVERDOSAGE": "overdosage",
    "DESCRIPTION": "description",
    "CLINICAL PHARMACOLOGY": "clinical_pharmacology",
    "MECHANISM OF ACTION": "mechanism_of_action",
    "PHARMACODYNAMICS": "pharmacodynamics",
    "PHARMACOKINETICS": "pharmacokinetics",
    "NONCLINICAL TOXICOLOGY": "nonclinical_toxicology",
    "CARCINOGENESIS, MUTAGENESIS, IMPAIRMENT OF FERTILITY": (
        "carcinogenesis_mutagenesis_impairment_of_fertility"
    ),
    "CLINICAL STUDIES": "clinical_studies",
    "HOW SUPPLIED/STORAGE AND HANDLING": "how_supplied_storage_and_handling",
    "PATIENT COUNSELING INFORMATION": "patient_counseling_information",
    "MEDICATION GUIDE": "medication_guide",
    "DRUG FACTS": "drug_facts",
    "ACTIVE INGREDIENT": "active_ingredient",
    "PURPOSE": "purpose",
    "USES": "uses",
    "DO NOT USE": "do_not_use",
    "ASK A DOCTOR BEFORE USE IF YOU HAVE": "ask_doctor_before_use",
    "ASK A DOCTOR OR PHARMACIST BEFORE USE IF YOU ARE": (
        "ask_doctor_or_pharmacist_before_use"
    ),
    "WHEN USING THIS PRODUCT": "when_using_this_product",
    "STOP USE AND ASK A DOCTOR IF": "stop_use_and_ask_doctor",
    "IF PREGNANT OR BREAST-FEEDING": "pregnancy_or_breastfeeding_otc",
    "KEEP OUT OF REACH OF CHILDREN": "keep_out_of_reach_of_children",
    "DIRECTIONS": "directions",
    "OTHER INFORMATION": "other_information",
    "INACTIVE INGREDIENTS": "inactive_ingredients",
    "QUESTIONS OR COMMENTS": "questions_or_comments",
    "RENAL IMPAIRMENT": "renal_impairment",
    "HEPATIC IMPAIRMENT": "hepatic_impairment",
    "POSTMARKETING EXPERIENCE": "postmarketing_experience",
    "POST-MARKETING EXPERIENCE": "postmarketing_experience",
    "CLINICAL TRIALS EXPERIENCE": "clinical_trials_experience",
    "REFERENCES": "references",
    "LABOR AND DELIVERY": "labor_and_delivery",
    "MICROBIOLOGY": "microbiology",
    "RISK SUMMARY": "risk_summary",
    "PRINCIPAL DISPLAY PANEL": "principal_display_panel",
    "PACKAGE LABEL.PRINCIPAL DISPLAY PANEL": "principal_display_panel",
    "PACKAGE/LABEL DISPLAY PANEL": "principal_display_panel",
    "GENERAL": "general_precautions",
    "LABORATORY TESTS": "laboratory_tests",
    "PATIENT INFORMATION": "patient_package_insert",
    "DRUG ABUSE AND DEPENDENCE": "drug_abuse_and_dependence",
    "CONTROLLED SUBSTANCE": "controlled_substance",
    "ABUSE": "abuse",
}

RETRIEVAL_FAMILY_SECTIONS: set[str] = {
    # Prescription label major sections
    "boxed_warning",
    "recent_major_changes",
    "indications_and_usage",
    "dosage_and_administration",
    "dosage_forms_and_strengths",
    "contraindications",
    "warnings_and_precautions",
    "warnings",
    "precautions",
    "adverse_reactions",
    "drug_interactions",
    "use_in_specific_populations",
    "overdosage",
    "description",
    "clinical_pharmacology",
    "nonclinical_toxicology",
    "clinical_studies",
    "how_supplied_storage_and_handling",
    "patient_counseling_information",
    "medication_guide",
    "patient_package_insert",
    "drug_abuse_and_dependence",

    # OTC Drug Facts families
    "drug_facts",
    "active_ingredient",
    "purpose",
    "uses",
    "directions",
    "do_not_use",
    "ask_doctor_before_use",
    "ask_doctor_or_pharmacist_before_use",
    "when_using_this_product",
    "stop_use_and_ask_doctor",
    "pregnancy_or_breastfeeding_otc",
    "keep_out_of_reach_of_children",
    "other_information",
    "inactive_ingredients",
    "questions_or_comments",
    "instructions_for_use",
}


def normalize_section_title(title: str | None) -> str | None:
    """
    Normalize an SPL section title for fallback matching.

    Examples:
      "1 INDICATIONS AND USAGE" -> "INDICATIONS AND USAGE"
      "8.1 Pregnancy" -> "PREGNANCY"
      "DO NOT USE " -> "DO NOT USE"
      "QUESTIONS OR COMMENTS?" -> "QUESTIONS OR COMMENTS"
    """
    if title is None:
        return None

    cleaned = " ".join(title.split()).strip()

    # Remove numeric PLR prefixes such as "8.1 " or "12.3 ".
    cleaned = re.sub(r"^\d+(?:\.\d+)*\s+", "", cleaned)

    # Remove common bullet/private-use glyphs that appear in label titles.
    cleaned = cleaned.replace("\uf0b7", "")
    cleaned = cleaned.replace("", "")

    # Normalize trailing punctuation that should not affect section identity.
    cleaned = cleaned.strip(" \t\r\n:;?")

    return cleaned.upper() or None


def map_canonical_section(
    loinc_code: str | None,
    raw_title: str | None,
) -> tuple[str | None, str]:
    """
    Return:
      (canonical_section_name, mapping_method)

    mapping_method is one of:
      - "loinc"
      - "title"
      - "unmapped"
    """
    if loinc_code and loinc_code in LOINC_TO_CANONICAL_SECTION:
        return LOINC_TO_CANONICAL_SECTION[loinc_code], "loinc"

    normalized_title = normalize_section_title(raw_title)
    if normalized_title and normalized_title in TITLE_TO_CANONICAL_SECTION:
        return TITLE_TO_CANONICAL_SECTION[normalized_title], "title"

    return None, "unmapped"