# ===============================
# ===== Makefile parameters =====
# ===============================

# fichiers sources
SRCDIR = src
# nom de l'executable produit
OUTPUT = LeCNN
# compilateur utilisé
CC = g++
# options de compilation pour la version de production
PRODFLAGS = -Ofast -flto -std=c++20 -Wall -pedantic
# options de compilation pour la version de debug
DEBUGFLAGS = -g -std=c++20 -Wall -pedantic

# ==============================
# ===== Makefile internals =====
# ==============================

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJDIR = .obj
OBJS=$(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
DBOBJS=$(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.do)
BINDIR = bin

.PHONY: all
all: $(BINDIR)/$(OUTPUT) $(BINDIR)/$(OUTPUT).db

$(BINDIR)/$(OUTPUT): $(OBJS) | $(BINDIR)
	$(CC) -o $@ $(PRODFLAGS) $^ -lm

$(BINDIR)/$(OUTPUT).db: $(DBOBJS) | $(BINDIR)
	$(CC) -o $@ $(DEBUGFLAGS) $^ -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CC) -o $@ -c $(PRODFLAGS) -I $(SRCDIR) $<

$(OBJDIR)/%.do: $(SRCDIR)/%.cpp | $(OBJDIR)
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