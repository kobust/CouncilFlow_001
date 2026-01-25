# Re-ranking vs. Context Reduction Analysis

## Current Setup (Re-ranking Disabled)

- **Retrieval**: `DEFAULT_TOP_K = 35` chunks per library
- **Over-retrieval**: `RETRIEVE_FACTOR = 2` → retrieves 70 chunks
- **Final context**: All 70 chunks included (after deduplication)
- **Re-ranking**: Disabled
- **LLM calls for retrieval**: 0 (just embedding + BM25)

## With Re-ranking Enabled

- **Retrieval**: Still retrieves 70 chunks (35 * 2)
- **Re-ranking**: LLM scores chunks in batches of 15
  - 70 chunks ÷ 15 = ~5 LLM calls
  - Each call: query + 15 chunks (800 chars each) ≈ 15-20k tokens
  - Total re-ranking tokens: ~75-100k tokens
- **Final context**: Top 35 chunks (after re-ranking)
- **Token savings**: 35 fewer chunks = ~15,750 tokens saved per call

## The Trade-off

**Re-ranking adds**: ~75-100k tokens (5 LLM calls for scoring)
**Re-ranking saves**: ~15,750 tokens (fewer chunks in final context)

**Net result**: Re-ranking INCREASES total token usage by ~60-85k tokens per retrieval

## Better Strategy: Reduce Context + Re-ranking

If you reduce `DEFAULT_TOP_K` from 35 to 20:

### Without Re-ranking:
- Retrieves: 40 chunks (20 * 2)
- Final context: 40 chunks
- Tokens: ~18,000 tokens in context

### With Re-ranking:
- Retrieves: 40 chunks (20 * 2)  
- Re-ranking: ~3 LLM calls (40 ÷ 15)
- Final context: 20 chunks (top-ranked)
- Re-ranking tokens: ~45-60k tokens
- Context tokens: ~9,000 tokens
- **Total**: ~54-69k tokens

### Without Re-ranking, smaller top_k:
- Retrieves: 40 chunks
- Final context: 40 chunks
- **Total**: ~18,000 tokens

## Conclusion

**Re-ranking INCREASES token usage** because:
1. The LLM calls for re-ranking use more tokens than the chunks they eliminate
2. Re-ranking is designed for precision, not token reduction

**Better approach for token reduction**:
1. **Reduce `DEFAULT_TOP_K`** (e.g., 35 → 20-25)
2. **Keep re-ranking disabled** (saves LLM calls)
3. **Hybrid retrieval (BM25 + semantic) already provides good precision** without re-ranking

**Re-ranking is only beneficial if**:
- You need maximum precision and accuracy is more important than token usage
- You're willing to trade ~60-85k extra tokens per retrieval for better chunk selection
- Your knowledge base is very large and you need to filter from many candidates

## Recommendation

**For token optimization**: 
- ❌ **Don't enable re-ranking** - it increases token usage
- ✅ **Reduce `DEFAULT_TOP_K`** to 20-25 chunks
- ✅ **Keep hybrid retrieval** (BM25 + semantic) - already provides good precision
- ✅ **Use delays** to spread token usage across time

**For maximum accuracy** (if tokens aren't a concern):
- ✅ Enable re-ranking
- ✅ Keep `DEFAULT_TOP_K` at 35 or higher
- ✅ You'll get better precision but use more tokens
