# Embedding Usage Analysis & Cost Optimization

## Current Embedding Usage

### ✅ Well Optimized

1. **Document Embeddings (Chunks)**
   - Embedded once during index build
   - Cached to disk via `rag_cache.py`
   - Reused on service restart
   - **Cost**: One-time per document chunk (very efficient)

2. **Query Embedding Reuse in Multi-Query Mode**
   - Queries are embedded once and reused across all libraries
   - Example: 3 queries × 3 libraries = 3 embedding calls (not 9)
   - **Good optimization already in place**

### ❌ Not Optimized

1. **Query Embeddings Not Cached**
   - Same query text embedded multiple times across:
     - Main prompt execution
     - Follow-on prompts (if similar queries)
     - Legal expert consultation (if similar queries)
   - **Cost**: Redundant API calls for identical queries

2. **No Session-Level Query Cache**
   - If user runs similar prompts, queries are re-embedded
   - No reuse of embeddings within a session

## Cost Analysis

### Embedding API Costs (Gemini)
- **Document embeddings**: ~$0.0001 per 1K characters (very cheap)
- **Query embeddings**: Same rate, but queries are short
- **Cost per query**: ~$0.00001-0.0001 (negligible per call)

### GenerateContent API Costs
- **Much more expensive**: ~$0.075-0.15 per 1M input tokens
- **This is where the real cost is**

## Optimization Opportunities

### 1. **Add Query Embedding Cache** (Low Impact, Easy Win)
- Cache query embeddings in session state
- Key: hash of query text
- Saves redundant embedding calls within a session
- **Impact**: Small cost savings, but good practice

### 2. **Batch Query Embeddings** (Already Done)
- ✅ Multi-query mode already batches and reuses
- No additional optimization needed

### 3. **Reduce Query Expansion** (High Impact)
- If `USE_QUERY_EXPANSION = True`, generates 3-5 queries
- Each query = 1 embedding call
- **Recommendation**: Disable if token limits are the issue (embeddings are cheap, but it adds LLM calls for expansion)

### 4. **Document Embedding Optimization** (Already Optimal)
- ✅ Already cached to disk
- ✅ Batched (100 per batch)
- ✅ Only done once per document

## Real Cost Drivers

**The main costs are NOT embeddings** - they're:
1. **GenerateContent calls** (main prompts, follow-ons, legal expert)
2. **Context caching** (one-time per run, but large)
3. **Query expansion LLM calls** (if enabled)

Embeddings are a tiny fraction of total cost.

## Recommendations

### For Cost Reduction (Focus Here):
1. ✅ **Reduce `DEFAULT_TOP_K`** (already done: 35 → 25)
2. ✅ **Add delays** (already done)
3. ⚠️ **Disable query expansion** (`USE_QUERY_EXPANSION = False`)
4. ⚠️ **Reduce context size** (smaller chunks or less overlap)

### For Embedding Optimization (Nice to Have):
1. Add query embedding cache (session state)
2. Minimal cost impact, but good practice

## Conclusion

**Embeddings are already well-optimized** for documents. Query embeddings could be cached, but the cost savings would be minimal. The real cost drivers are GenerateContent calls, which we've already optimized with delays and reduced retrieval size.
