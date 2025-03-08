# ===============================
# ===== Makefile parameters =====
# ===============================

# fichiers sources
SRCDIR = src
# nom de l'executable produit
OUTPUT = LeCNN
# compilateur utilisé
CC = gcc
# options de compilation pour la version de production
PRODFLAGS = -O3 -flto -std=c11 -Wall -pedantic
# options de compilation pour la version de debug
DEBUGFLAGS = -g -std=c11 -Wall -pedantic

# ==============================
# ===== Makefile internals =====
# ==============================

SRCS = $(wildcard $(SRCDIR)/*.c)
OBJDIR = .obj
OBJS=$(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DBOBJS=$(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.do)
BINDIR = bin

.PHONY: all
all: clean $(BINDIR)/$(OUTPUT) $(BINDIR)/$(OUTPUT).db

$(BINDIR)/$(OUTPUT): $(OBJS) | $(BINDIR)
	$(CC) -o $@ $(PRODFLAGS) $^ -lm

$(BINDIR)/$(OUTPUT).db: $(DBOBJS) | $(BINDIR)
	$(CC) -o $@ $(DEBUGFLAGS) $^ -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) -o $@ -c $(PRODFLAGS) -I $(SRCDIR) $<

$(OBJDIR)/%.do: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) -o $@ -c $(DEBUGFLAGS) -I $(SRCDIR) $<

.PHONY: $(OBJDIR)
$(OBJDIR):
	@mkdir -p $@

.PHONY: $(BINDIR)
$(BINDIR):
	@mkdir -p $@
	
.PHONY: clean
clean:
	rm -f $(BINDIR)/$(OUTPUT) $(BINDIR)/$(OUTPUT).db $(OBJDIR)/*.o $(OBJDIR)/*.do
	rmdir $(OBJDIR) 2>/dev/null || true
	rmdir $(BINDIR) 2>/dev/null || true