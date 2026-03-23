results.mk:
	@{ \
	  if [ -n "$$TK_MATHLIBS_CFLAGS" ]; then \
	    cflags="$$TK_MATHLIBS_CFLAGS"; \
	  elif pkg-config --cflags blas lapack lapacke 2>/dev/null 1>/dev/null; then \
	    cflags="$$(pkg-config --cflags blas lapack lapacke)"; \
	  elif command -v brew >/dev/null 2>&1; then \
	    cflags=""; \
	    for lib in openblas lapack; do \
	      p=$$(brew --prefix $$lib 2>/dev/null) && \
	        cflags="$$cflags -I$$p/include"; \
	    done; \
	    p=$$(brew --prefix libomp 2>/dev/null) && \
	      cflags="$$cflags -I$$p/include"; \
	  fi; \
	  if [ -n "$$TK_MATHLIBS_LDFLAGS" ]; then \
	    ldflags="$$TK_MATHLIBS_LDFLAGS"; \
	  elif pkg-config --libs blas lapack lapacke 2>/dev/null 1>/dev/null; then \
	    ldflags="$$(pkg-config --libs blas lapack lapacke)"; \
	  elif command -v brew >/dev/null 2>&1; then \
	    ldflags=""; \
	    for lib in openblas lapack; do \
	      p=$$(brew --prefix $$lib 2>/dev/null) && \
	        ldflags="$$ldflags -L$$p/lib"; \
	    done; \
	    p=$$(brew --prefix libomp 2>/dev/null) && \
	      ldflags="$$ldflags -L$$p/lib"; \
	    ldflags="$$ldflags -lopenblas -llapacke -lomp"; \
	  else \
	    ldflags="-lopenblas -llapacke"; \
	  fi; \
	  case "$$(uname -s)" in \
	    Linux*) ldflags="$$ldflags -Wl,-z,nodelete" ;; \
	  esac; \
	  echo "MATHLIBS_CFLAGS = $$cflags"; \
	  echo "MATHLIBS_LDFLAGS = $$ldflags"; \
	} > $@
