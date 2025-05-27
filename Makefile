# ===============================
# ===== Makefile parameters =====
# ===============================

# fichiers sources
SRCDIR = src
# nom de l'executable produit
OUTPUT = network
# compilateur utilisé
CC = gcc
# options de compilation pour la version de production
PRODFLAGS = -Ofast -flto -std=c11 -Wall -Wextra -pedantic
# options de compilation pour la version de debug
DEBUGFLAGS = -g -std=c11 -Wall -Wextra -pedantic

# ==============================
# ===== Makefile internals =====
# ==============================

SRCS = $(wildcard $(SRCDIR)/*.c)
OBJDIR = .obj
OBJS=$(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DBOBJS=$(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.do)
BINDIR = bin

release: $(BINDIR)/$(OUTPUT)

debug: $(BINDIR)/$(OUTPUT).db

$(BINDIR)/$(OUTPUT): $(OBJS) | $(BINDIR)
	$(CC) -o $@ $(PRODFLAGS) $^ -lm

$(BINDIR)/$(OUTPUT).db: $(DBOBJS) | $(BINDIR)
	$(CC) -o $@ $(DEBUGFLAGS) $^ -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) -o $@ -c $(PRODFLAGS) -I $(SRCDIR) -MD -MP -MF $(OBJDIR)/$*.d $<

$(OBJDIR)/%.do: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) -o $@ -c $(DEBUGFLAGS) -I $(SRCDIR) -MD -MP -MF $(OBJDIR)/$*.dd $<

-include $(OBJDIR)/*.d $(OBJDIR)/*.dd

$(OBJDIR):
	@mkdir -p $@

$(BINDIR):
	@mkdir -p $@

clean:
	rm -f $(BINDIR)/$(OUTPUT) $(BINDIR)/$(OUTPUT).db $(OBJDIR)/*.o $(OBJDIR)/*.do $(OBJDIR)/*.d $(OBJDIR)/*.dd
	rmdir $(OBJDIR) 2>/dev/null || true
	rmdir $(BINDIR) 2>/dev/null || true

.PHONY: release debug $(OBJDIR) $(BINDIR) clean