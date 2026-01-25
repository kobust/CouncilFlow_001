# Token Usage Optimization Strategies

This document outlines strategies to reduce Gemini API token usage while maintaining accuracy.

## Current Token Usage Points

1. **Main prompt execution** - Uses cached context (efficient)
2. **Query expansion** (if enabled) - 1 LLM call per run
3. **Retrieval planner** (if enabled) - 1 LLM call per run  
4. **Legal expert consultation** - Separate RAG search + LLM call
5. **Follow-on prompts** - Each does its own RAG search + LLM call
6. **Re-ranking** (if enabled) - Multiple LLM calls in batches

## Optimization Strategies

### 1. **Increase Pace Delays** (Immediate Impact)
- **Current**: `GEMINI_PACE_DELAY_SECONDS=0` (no delay)
- **Recommendation**: Set to 15-30 seconds between GenerateContent calls
- **Impact**: Spreads token usage across time, reducing per-minute spikes
- **Trade-off**: Slightly slower execution, but prevents quota errors

### 2. **Disable Optional LLM Features** (High Impact)
- **Query Expansion**: Set `USE_QUERY_EXPANSION = False` in `rag_loader.py`
  - Saves 1 LLM call per prompt execution
  - Falls back to single search phrase (still accurate with hybrid retrieval)
- **Retrieval Planner**: Already disabled (`USE_RETRIEVAL_PLANNER = False`)
- **Re-ranking**: Already disabled (`RERANK_ENABLED = False`)

### 3. **Reduce Retrieval Size** (Moderate Impact)
- **Current**: `DEFAULT_TOP_K = 35` chunks per library
- **Recommendation**: Reduce to 20-25 chunks
- **Impact**: Smaller context = fewer tokens per call
- **Trade-off**: Might miss some relevant chunks, but hybrid retrieval helps

### 4. **Add Delays Between Pipeline Steps** (High Impact)
- Add delays between follow-on prompts
- Add delays before legal expert consultation
- **Impact**: Prevents burst of API calls in quick succession

### 5. **Use Lighter Models for Auxiliary Tasks** (Moderate Impact)
- Use `gemini-2.0-flash` for query expansion (already using PLANNER_MODEL)
- Keep main analysis on `gemini-3-flash-preview` for accuracy
- **Impact**: Lower token costs for auxiliary calls

### 6. **Optimize Context Size** (Moderate Impact)
- Reduce chunk size slightly (currently 1800 chars)
- Reduce overlap (currently 220 chars)
- **Trade-off**: Might reduce context continuity

### 7. **Cache Reuse** (Already Optimized)
- Context caching is already implemented
- Legal expert and follow-on prompts create new caches (necessary for different queries)

### 8. **Batch Legal Expert Questions** (Future Enhancement)
- If multiple prompts have legal questions, batch them into one consultation
- **Impact**: Reduces legal expert calls

## Recommended Immediate Actions

### Quick Wins (Implemented)

✅ **Inter-step delays**: Now configurable via `PIPELINE_STEP_DELAY_SECONDS` (default: 10s)
✅ **Legal expert delays**: Now configurable via `LEGAL_EXPERT_DELAY_SECONDS` (default: 10s)
✅ **Pace delays**: Already available via `GEMINI_PACE_DELAY_SECONDS`

### Additional Optimizations (Manual Configuration)

1. **Increase pace delay**: Set `GEMINI_PACE_DELAY_SECONDS=20` (or higher)
   - Current default: 0 seconds
   - Recommendation: 20-30 seconds for heavy usage
   - Impact: Spreads token usage across time, prevents per-minute spikes

2. **Disable query expansion**: Set `USE_QUERY_EXPANSION = False` in `rag_loader.py`
   - Current: Enabled (1 extra LLM call per prompt)
   - Impact: Saves 1 LLM call per prompt execution
   - Trade-off: Uses single search phrase instead of 3-5, but hybrid retrieval maintains accuracy

3. **Reduce retrieval size**: Set `DEFAULT_TOP_K = 25` in `rag_loader.py` (from 35)
   - Impact: ~30% reduction in context tokens per call
   - Trade-off: Fewer chunks, but hybrid retrieval + semantic search maintains relevance

4. **Increase pipeline delays**: Set `PIPELINE_STEP_DELAY_SECONDS=15` (from 10)
   - Impact: More time between follow-on prompts = lower per-minute token usage

5. **Increase legal expert delays**: Set `LEGAL_EXPERT_DELAY_SECONDS=15` (from 10)
   - Impact: Spreads legal consultation calls across time

## Accuracy Preservation

- Hybrid retrieval (BM25 + semantic) maintains accuracy even with fewer chunks
- Context caching ensures knowledge base is efficiently reused
- Legal expert consultation still gets full context for its specialized search
- Follow-on prompts re-plan retrieval, ensuring relevant context
