A running outline of my thoughts so far on how to tell the contrastVI story/what our
figures should probably look like.

---

### Thoughts on Main Text Figures:

Figure 1: Concept figure + application.
* I think we should model our Figure 1 based off the Figure 1 in the scArches paper
* At the top, we'll have a concept (sub)figure 
* At the bottom, we can then provide some results for our initial AML experiments
(PCA/UMAP plots, as well as some basic quantitative results (ARI/Silhouette/etc))

Figure 2: Application to Epithelial cell data

* Here we can showcase both that infection signal is encoded in the salient space, and
that common signals (e.g. cell type differences) are encoded in the background space
* Some visualizations (UMAP) will be good here, along with some more quantitative results

Figure 3: Cross-species application

* Here we could use a simpler species (Mice?) as a reference for more complicated
species (Humans?). Ideally, we'd see human-specific stuff become more obvious in the
salient latent space.

Figure 4: ???

Figure 5: ???

---

### Thoughts on supplemental figures

* Sensitivity analysis (e.g. how does size of salient latent space change quality of
embeddings?)
* Runtime analysis
    * How much slower are we than scVI?
    * How well do we scale as # of cells increases?
* Something about the likelihood bound
    * How much worse/better is our log-likelihood compared to standard scVI?
* Plots for baseline methods
    * Assuming these don't make it into the main text, we should throw UMAP/PCA plots for
    baseline methods here
* Some kind of robustness analysis
    * What happens if we e.g. hold out one cell type from the target dataset during training
    * Do we still get decent results?
* Something about analysis which genes are most affected by the salient latent space
    * Compare what happens when we decode using only background variables vs when we use
    both background + salient variables
