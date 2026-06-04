# Backlog

- Additional datasets (AmazonCat-13K, SNLI, QQP)

## Tokenizer input validation / review (prevent segfaults + silent misuse)

- tokenize_annotated: validate dimensions — span_starts/ends/types length vs
  doc_span_offsets[n_docs], and doc_span_offsets length vs #texts. Currently
  unchecked -> OOB/segfault on mismatch.
- tokenize_annotated: validate span_types are in [0, n_types); negative type ids
  index safe_open/safe_close out of bounds. Same class of missing input checks.
- tokenize: unrecognized `collapse` string silently falls back to COL_NONE;
  a typo gives "none" with no error. Validate/error on unknown collapse.
- tokenize: `n_samples` must match the texts/sequences count; a too-large value
  silently yields trailing empty rows. Validate or warn.
- tokenize/tokenize_annotated: control-byte sentinels (focus \x01, open/close
  0x02-0x1E, terminals \x03/\x04) collide if input text contains those bytes.
  Consider making the sentinel set / escaping configurable.
- terminals is ignored for `sequences` input (sequences path returns before the
  terminals block). Users can prepend sentinels themselves; revisit only if
  needed. (RESOLVED as won't-fix per discussion.)
