# Text Sequence Debugging Status – Digital Twin Pipeline

## 1. Problem Statement

The core issue is that the **translated text in the output document does not match the original reading order of the source PDF**. This disrupts document coherence and usability.

## 2. Actions Taken So Far

- **Sequential Output Enforcement:**
  - Modified the pipeline to output blocks in the order they are extracted, rather than spatially re-sorting them.
- **Debug Exports Added:**
  - **Block Mapping Debug:** Exports the mapping of block IDs, types, original and translated text, batch IDs, and batch indices after translation.
  - **Final Output Order Debug:** Exports the order of blocks as they are written to the output document, including block IDs, types, and text.
- **Comparison of Orders:**
  - Compared extraction, translation, and output orders to pinpoint where the sequence diverges.

## 3. Purpose of Debugging Devices

- **Block Mapping Debug:**
  - To trace how each block moves through batching and translation, and to see if mapping is lost or segments are misassigned.
- **Final Output Order Debug:**
  - To verify the exact order in which blocks are written to the output document, and to compare this with extraction and translation mapping.
- **(Planned) Extraction Order Debug:**
  - To capture the order in which blocks are first extracted from the PDF, for a ground-truth reference.

## 4. Current Findings

- The output document and final output JSON now reflect the order of blocks as they are processed for output.
- However, the **block IDs in the output do not match the true reading order**, and the text sequence in the output is still not correct.
- The root cause may be in extraction, batching, translation mapping, or output logic.

## 5. Next Steps

1. **Export Extraction Order:**
   - Add a debug export immediately after extraction to capture the original block order.
2. **Compare All Orders:**
   - Compare extraction, translation, and output orders to pinpoint exactly where the sequence is lost.
3. **Fix the Pipeline:**
   - Once the divergence point is found, update the pipeline to preserve the original reading order through all stages.

---

**Goal:**
> Ensure that the translated text in the output document appears in the exact same reading order as the original PDF. 