Project: Multimodal RAG Assistant for Traffic Sign Understanding (SmartSign RAG)

Stack: LangChain, Multimodal Embeddings (CLIP), ChromaDB, Streamlit, Llama 3.


        1. Problem & Users

1.1 Problem Statement

Understanding traffic rules and identifying signs often requires consulting extensive text-based manuals. There is a disconnect between visual perception (seeing a sign) and its legal/technical description. Standard search engines often return either just an image or dry text without real-world visual examples.

1.2 Target Audience

 -Driving School Students: Users who need quick clarifications of signs with visual examples.
 -ADAS Developers (Advanced Driver Assistance Systems): As a support tool for verifying sign classes and their technical descriptions.
 -Data Scientists: Users needing an interface to verify the consistency between text descriptions and visual data in the GTSRB dataset.


        2. MVP Scope

2.1 In-Scope (Features)

 -Multimodal Retrieval: Finding text explanations and corresponding images via semantic queries (e.g., "signs that prohibit stopping").
 -Grounded Generation: LLM response generation based strictly on retrieved fragments of rules and descriptions.
 -Source Visualization: Displaying text citations and a gallery of retrieved thumbnails alongside the answer.
 -Provenance: Clear links to `class_id` and data sources for every part of the response.

2.2 Out-of-Scope

 -Agentic Loops: The system follows linear logic (Query -> Retrieval -> Context -> Answer).
 -Tool Use: No code execution or external API calls during query time.
 -Real-time Computer Vision: The system does not classify user-uploaded photos in real-time but searches an existing indexed database.


        3. Content & Data

3.1 Data Sources

 -Images: GTSRB (German Traffic Sign Recognition Benchmark). A sample of ~1500 images (multiple variants for each of the 43 classes).
 -Text: A structured corpus of descriptions (based on official traffic regulations). Each `class_id` includes:
    Official name.
    Detailed rule description.
    Category (Prohibitory, Mandatory, Warning, etc.).

3.2 Data Linkage

Data is linked via metadata: every text chunk and every image must have a mandatory `class_id` field.


        4. Example Queries

| Query Type                | Example |

| Image-Oriented            | "Show me what the 'Priority over oncoming traffic' sign looks like." |
| Image-Oriented            | "Find all images of speed limit signs above 60 km/h." |
| Image-Oriented            | "What do road work warning signs look like?" |
| Multimodal (Text + Photo) | "Explain the meaning of the sign with a deer and show examples of it on the road." |
| Analytical (Comparison)   | "What is the difference between 'No Stopping' and 'No Parking' signs?" |


        5. Success Metrics

 -Retrieval Relevance (R@5): The correct `class_id` must be present in the top-5 search results in >85% of cases.
 -Answer Faithfulness: 100% of claims in the LLM response must reference the provided context (evaluated via LLM-as-a-judge or manual audit).
 -Latency: Time from query input to response (including retrieval) < 4 seconds on local GPU/high-end CPU.


        6. Technical Choices

Orchestration: LangChain (LCEL for chain construction).
Vector Database: ChromaDB (for metadata support and multimodal vector integration).
Embeddings: CLIP (OpenAI or HuggingFace version) — to create a shared vector space for text and images.
LLM: Llama 3.
UI: Streamlit (Two-column interface: Chat + Source Gallery).

