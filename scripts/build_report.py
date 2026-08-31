"""Build the exact 27-page APA-style methodology report.

The report intentionally contains no fabricated empirical result. Figures are model,
design, privacy, and validation schematics generated from the frozen protocol.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "Encrypted_Protocol_Topology_Recovery_APA_Report.pdf"
REPO_COPY = ROOT / "report" / OUTPUT.name
FIGURES = ROOT / "report" / "figures"
REGULAR_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def make_figures() -> dict[str, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    # Exact likelihood decomposition.
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    names = ["Event log-intensity", "Baseline compensator", "Excitation compensator"]
    values = [1, -1, -1]
    bars = ax.barh(names, values, color=["#21b6d7", "#ffbe55", "#ff7b73"])
    ax.axvline(0, color="#243b4a", lw=1)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_title("All three terms are required by the point-process likelihood", weight="bold")
    for bar, label in zip(bars, ["Σ log λ", "−TΣμ", "−ΣA(1−e⁻ᵝᵘ)"], strict=True):
        x = bar.get_width()
        ax.text(x * 0.52, bar.get_y() + bar.get_height() / 2, label, ha="center", va="center")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    likelihood = FIGURES / "likelihood_decomposition.png"
    fig.savefig(likelihood, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Design matrix schematic.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})
    matrix = np.arange(9).reshape(3, 3)
    axes[0].imshow(matrix, cmap="Blues", alpha=0.85)
    axes[0].set_xticks(range(3), ["Dense", "Moderate", "Sparse"])
    axes[0].set_yticks(range(3), ["None", "Padding", "Jitter"])
    axes[0].set_xlabel("Graph sparsity")
    axes[0].set_ylabel("Obfuscation")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, "2 architectures", ha="center", va="center", fontsize=8)
    axes[0].set_title("18 cells per truth seed", weight="bold")
    axes[1].bar(["Blocks", "Cells/block", "Runs"], [100, 18, 1800], color=["#8b6ee8", "#38cbe8", "#5bd9a3"])
    axes[1].set_yscale("log")
    axes[1].set_title("Frozen allocation", weight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    design = FIGURES / "factorial_design.png"
    fig.savefig(design, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Privacy pipeline.
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    labels = ["Authorized\ninterface", "Six-field\nextractor", "HMAC\npseudonyms", "Schema\naudit", "Inference\nwindow"]
    xs = np.linspace(0.9, 9.1, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels, strict=True)):
        color = "#d9f6fb" if i != 2 else "#e9e0ff"
        ax.add_patch(plt.Rectangle((x - 0.72, 0.78), 1.44, 0.86, fc=color, ec="#28738b", lw=1.3))
        ax.text(x, 1.21, label, ha="center", va="center", weight="bold", fontsize=8)
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.78, 1.21), xytext=(x + 0.78, 1.21), arrowprops={"arrowstyle": "->", "color": "#28738b"})
    ax.text(5, 0.3, "Payload · raw addresses · ports · interface names · PCAP are excluded", ha="center", color="#9d382f", weight="bold")
    ax.set_title("Data minimization occurs before persistence", weight="bold")
    fig.tight_layout()
    privacy = FIGURES / "privacy_pipeline.png"
    fig.savefig(privacy, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Evidence ladder.
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    labels = ["Likelihood\nunit tests", "Flow and\nprivacy tests", "Controlled-truth\nfactorial", "Authorized live\nfeasibility", "Independent\noperational test"]
    state = [1, 1, 0, 0, 0]
    colors_ = ["#51d49d" if x else "#d9dfe4" for x in state]
    ax.bar(range(5), [1] * 5, color=colors_, edgecolor="#426071")
    ax.set_xticks(range(5), labels)
    ax.set_yticks([])
    ax.set_ylim(0, 1.25)
    ax.set_title("Evidence gates: implementation tests do not substitute for execution", weight="bold")
    for i, done in enumerate(state):
        ax.text(i, 0.5, "complete" if done else "pending", ha="center", va="center", weight="bold", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    ladder = FIGURES / "evidence_ladder.png"
    fig.savefig(ladder, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"likelihood": likelihood, "design": design, "privacy": privacy, "ladder": ladder}


PAGES: list[tuple[str, list[str]]] = [
    (
        "Abstract",
        [
            "Encrypted transport removes payload content from direct inspection but does not erase the temporal and relational structure of packet exchange. This report specifies a privacy-minimized statistical laboratory for estimating latent directed communication topology from authorized packet metadata. The observation consists only of event time, frame length, keyed endpoint pseudonyms, and coarse protocol families. Raw addresses, ports, payloads, interface names, and packet-capture files are excluded before persistence.",
            "The inferential architecture combines an exact exponential-kernel multivariate Hawkes likelihood, an amortized dynamic stochastic block model, and masked autoregressive variational flows. Unlike the preliminary formulation, the likelihood contains both the event log-intensity and its observation-window compensator; the variational density contains the flow Jacobian; excitation is constrained below a stability boundary; and optimization trajectories are not misrepresented as Markov chains. A static graph autoencoder with Louvain clustering supplies a deliberately simpler comparator.",
            "Validation separates a controlled-truth randomized complete-block experiment from unlabelled live feasibility. One hundred truth seeds cross two architectures, three obfuscation conditions, and three sparsity levels, giving 1,800 planned runs. Primary controlled-truth endpoints are link area under the receiver operating characteristic curve, binary link log score, and compute latency per event. The repository is methodology-ready but contains no live capture and no completed factorial result; therefore no empirical superiority, attribution, or operational cyber-signals claim is made.",
            "Keywords: Hawkes process, dynamic stochastic block model, normalizing flow, packet metadata, variational inference, topology recovery, data minimization",
        ],
    ),
    (
        "Encrypted Metadata as a Statistical Observation",
        [
            "Modern encrypted protocols motivate a distinction between content and structure. Payload encryption protects message content, whereas event timing, frame length, and endpoint interaction may remain observable at an authorized receiver. The statistical problem is not decryption. It is estimation of a latent directed graph whose edges encode probabilistic communication association among pseudonymous nodes. That distinction is foundational: the estimand is relational intensity, not semantic intent or human identity.",
            "Let the persisted observation be a marked point pattern on a bounded interval. Each event contains a time, a directed pseudonymous dyad, a frame-length mark, and two coarse protocol-family marks. The generated sigma-field excludes payload bytes, names, ports, raw addresses, hardware addresses, and interface identity. Every inferential statement is consequently conditional on this reduced observation. No downstream model can legitimately restore excluded meaning without additional evidence.",
            "The open-world difficulty comes from nonstationarity, excitation, and uncertain block structure. Bursty exchanges create history-dependent intensity; nodes can share latent roles; obfuscation perturbs timing or sizes; and the number of occupied blocks is not known in advance. A finite truncation is used computationally, but the model is designed so unused blocks may receive negligible posterior mass. The evaluation asks whether richer temporal inference improves recovery under controlled truth, not whether metadata alone proves a security narrative.",
        ],
    ),
    (
        "Observation Space and Measurability",
        [
            "Formally, events live on the product space of time, directed dyads, and marks. The counting measure records occurrences on this space, while the predictable intensity is measurable with respect to the left-continuous history filtration. Pseudonymization is a deterministic measurable map conditional on a locally held key. It preserves equality relationships needed for graph construction while suppressing direct address disclosure in persisted files.",
            "Data minimization changes the estimand rather than merely changing file format. Because ports and payload-derived protocol labels are omitted, a node represents an endpoint pseudonym at the chosen address layer, not an application, user, process, or device identity. Because interface names are omitted, released metadata cannot identify the physical receiver. Repeated runs using the same local key preserve within-study node correspondence; changing the key deliberately breaks that correspondence across studies.",
            "The observation window is finite and begins at its first retained event. Conditional likelihood calculations therefore inherit a boundary assumption about unobserved prehistory. The implementation treats the first event as the window origin and does not fabricate earlier events. Sensitivity analysis can discard an initial warm-up segment or initialize excitation from a stationary approximation. Those choices must be frozen before outcome inspection because they influence early-window intensity and apparent burstiness.",
        ],
    ),
    (
        "Exact Multivariate Hawkes Likelihood",
        [
            "For event type r, the conditional intensity equals a positive baseline plus exponentially decayed contributions from earlier event types. The log likelihood on a finite interval is the sum of observed event log-intensities minus the integrated intensity over the entire event-type domain. This compensator is the normalization term induced by the point-process measure; omitting it rewards arbitrarily large intensity and does not define the intended likelihood.",
            "With an exponential kernel, the compensator has a closed form. The baseline contribution is the window length times the sum of baselines. Each earlier event contributes its excitation coefficient multiplied by one minus an exponential tail factor. The implementation evaluates these terms with positive transformed parameters and finite-value checks. Unit tests compare analytic values to direct numerical constructions and require finite gradients.",
            "The preliminary code accumulated only logarithms of event-time intensities and added a small constant before the logarithm. The corrected engine instead obtains strict positivity by softplus parameterization and a declared numerical floor used solely at machine precision. This is not an ad hoc likelihood component. The exact decomposition is visible in Figure 1, which emphasizes that event reward and integrated exposure are inseparable.",
        ],
    ),
    (
        "Marked Process Extension",
        [
            "Frame length is treated as a mark conditional on the latent block relation, while the primary event-time model remains a multivariate counting process. The current implementation uses a regularized Gaussian mark contribution after scale stabilization. This factorization states an explicit conditional independence assumption: given latent roles and dyadic event type, size marks do not alter the event-time compensator. A richer joint marked intensity may be substituted only with a new normalization derivation.",
            "Padding attacks principally affect the mark channel. Deterministic common-size padding removes information in size dispersion without directly changing event times. Jitter attacks principally affect the time channel by perturbing arrival locations, potentially attenuating apparent excitation or reversing near-simultaneous ordering. The factorial design separates these mechanisms rather than collapsing them into a single difficulty score.",
            "Live traces can contain retransmissions, batching, and receiver artifacts that violate conditional mark assumptions. Accordingly, mark fit is secondary to controlled link recovery, and live evaluation emphasizes posterior sensitivity across windows. A posterior predictive check should compare size quantiles, interarrival distributions, dyadic counts, and burst durations without exposing individual pseudonyms in public artifacts.",
        ],
    ),
    (
        "Stability and Identifiability",
        [
            "A stationary linear Hawkes process requires the excitation operator to remain below its critical branching boundary. The software rescales the nonnegative excitation matrix to a spectral-norm ceiling of 0.85. Spectral norm is a conservative computational constraint relative to the exact spectral radius, but it is differentiable almost everywhere and ensures the fitted process does not cross the instability boundary under the implemented parameterization.",
            "Topology and excitation are only partially identifiable from short windows. A high baseline with weak excitation can resemble a lower baseline with stronger residual history, while block mixing can exchange labels without changing the likelihood. Block-label switching is addressed by permutation-invariant link estimands, not by claiming semantic labels. Posterior summaries should never interpret block number as an intrinsic category.",
            "Identifiability also depends on excitation diversity. If every dyad exhibits the same event rate and kernel response, the block structure receives little information. The controlled simulator varies within-block and between-block propensities while fixing the target graph for paired obfuscation conditions. This permits recovery evaluation against known adjacency without assuming that real endpoints possess externally meaningful classes.",
        ],
    ),
    (
        "Dynamic Stochastic Block Structure",
        [
            "The block model represents each node by a probability vector over a finite truncation of latent roles. Directed block-pair logits generate link probabilities, allowing asymmetric association. Relaxed Gumbel–Softmax draws make the discrete allocation differentiable during optimization. The truncation bounds computation; it does not assert that the true open-world process contains exactly that many roles.",
            "Node features summarize only authorized metadata: outgoing and incoming counts, size moments, and timing-rate statistics. An amortized encoder maps these features to variational parameters. The resulting node embeddings enter both the relaxed block allocation and the directed link decoder. Temporal evolution is handled through repeated windows and updated histories, whereas the controlled factorial evaluates a frozen window to isolate architecture effects.",
            "Dynamic stochastic block modeling is used here as a structural abstraction, not a semantic classifier. A block may capture traffic regularity, receiver aggregation, or a transient protocol pattern. Interpretation requires external validation. The defensible output is a posterior distribution over relational structure and occupied latent components among pseudonymous endpoints.",
        ],
    ),
    (
        "Masked Autoregressive Variational Flow",
        [
            "A diagonal Gaussian posterior is often too rigid for multimodal graph allocations. The masked autoregressive flow transforms Gaussian base draws through a sequence of triangular bijections. Each coordinate shift and scale depends only on preceding coordinates under a fixed ordering. This autoregressive masking yields a tractable triangular Jacobian, so the transformed log density is evaluated exactly for each Monte Carlo draw.",
            "Smooth bounded log-scales prevent numerical explosion while preserving invertibility. Alternating coordinate order between layers improves dependence coverage. The implementation provides both forward and inverse maps; round-trip tests require reconstruction to numerical tolerance and opposing log-determinants. This directly tests the mathematical object whose Jacobian enters the evidence lower bound.",
            "The flow increases variational flexibility but does not make the posterior exact. Approximation quality is assessed with held-out predictive scores, sensitivity across initializations, and self-normalized importance weights. Importance effective sample size describes weight degeneracy for this approximation only. It is not the autocorrelation-based effective sample size of a Markov chain and is not a fair universal endpoint against a deterministic static optimizer.",
        ],
    ),
    (
        "Evidence Lower Bound and Optimization",
        [
            "The evidence lower bound is the Monte Carlo expectation of the joint log density minus the variational log density. The joint contribution combines the exact event-time likelihood, mark likelihood, block-structured graph prior, and regularizing priors. The variational term includes the Gaussian base density and every flow log-Jacobian. Omitting a Jacobian would optimize a different, generally invalid objective.",
            "Optimization uses Adam with gradient clipping and finite-value guards. Device synchronization surrounds only measured compute segments when CUDA is available; acquisition delay and file input/output are excluded. The final objective per event is a training diagnostic rather than a model-comparison score because the two architectures optimize different objectives. Confirmatory comparisons use common controlled-truth endpoints.",
            "Variational restarts are not Markov chains. Consequently, the repository does not compute Gelman–Rubin statistics from epochs, restarts, or minibatches. Rank-normalized folded R-hat is retained as a tested utility only for future algorithms that actually produce independent chains. This separation repairs a category error in the original specification.",
        ],
    ),
    (
        "Static Comparator",
        [
            "The comparator aggregates directed event counts into a static graph, encodes node features with a graph autoencoder, and decodes pairwise links from latent inner products. Louvain clustering supplies a heuristic partition for descriptive comparison. This model ignores continuous event time and therefore cannot represent excitation or the Hawkes compensator.",
            "A simpler comparator remains scientifically useful because it tests whether temporal modeling adds value beyond aggregated connectivity. It is not described as a second Bayesian posterior, and no artificial MCMC diagnostic is attached. Both methods return directed link scores on the same controlled truth graph, enabling link AUC and log-score contrasts.",
            "Implementation parity is pursued through identical truth seeds, obfuscation transformations, optimization budgets recorded in metadata, and synchronized latency measurement. Absolute parameter counts may differ; the report therefore includes parameter and convergence summaries as descriptive covariates. Any superiority statement must be restricted to the frozen compute environment and experimental support.",
        ],
    ),
    (
        "Authorized Live Capture Architecture",
        [
            "The live lane offers two acquisition backends. Scapy receives packet objects from a single named interface with storage and promiscuous mode disabled. TShark emits only selected fields to standard output and never receives a packet-output path. Both routes transform endpoints with a local keyed hash before writing a compressed metadata block.",
            "The command line requires an explicit authorization assertion. This is an operational guardrail rather than a legal determination: the operator remains responsible for permission, institutional policy, and applicable law. The software does not attempt to evade operating-system permissions, endpoint controls, or organizational application-control policy. If capture or a signed library is blocked, the correct response is administrative approval or an approved environment.",
            "Live processing can run immediately after a bounded capture, but the shipped repository contains no packet trace. This avoids publishing potentially identifying relational data. Public demonstrations use only the controlled simulator or browser-native didactic graphs whose status is clearly labeled.",
        ],
    ),
    (
        "Privacy Engineering and Data Minimization",
        [
            "Data minimization is enforced structurally. Persisted rows contain time, frame length, two endpoint pseudonyms, and two coarse protocol families. Payload, ports, raw IP and MAC addresses, DNS names, TCP sequence fields, interface names, and packet-capture files are absent. A schema audit rejects forbidden keys, and unit tests verify the permitted field set.",
            "Endpoint pseudonyms are HMAC-SHA256 values under a locally generated 256-bit salt. Unlike an unsalted digest, keyed hashing resists straightforward enumeration when the address candidate set is small. Nevertheless, pseudonymization is not anonymization: degree, timing, and repeated relational patterns may remain identifying. Capture blocks are therefore ignored by Git and intended for local aggregate analysis.",
            "Atomic gzip JSON writes reduce partial-file risk, and each block carries a SHA-256 integrity digest over canonicalized content. The salt is stored separately under a private ignored path. Figure 2 shows that minimization happens before persistence and model fitting, not as a post hoc redaction step.",
        ],
    ),
    (
        "Ethical and Operational Boundary",
        [
            "The research purpose is statistical method evaluation on networks the operator is authorized to observe. The repository is not designed for covert collection, credential discovery, payload reconstruction, decryption, or targeting. It does not infer human identity, malign intent, organizational ownership, or command-and-control status. Such claims require independent legal, technical, and contextual evidence outside this model.",
            "Risk remains even when payloads are absent. Relational metadata can expose behavior, routines, and associations. The protocol therefore minimizes fields, bounds duration and packet count, avoids public raw data, and recommends aggregate release only. Formal institutional review may be necessary whenever data involve people, organizations, or networks beyond a purely personal test environment.",
            "The Menlo principles of respect for persons, beneficence, justice, and respect for law and public interest offer a useful frame for information-and-communications-technology research. In this project they motivate explicit permission, minimal collection, proportional retention, honest uncertainty, and a public claim boundary that is narrower than the engineering capability.",
        ],
    ),
    (
        "Controlled-Truth Simulator",
        [
            "Controlled truth is essential because a live encrypted network rarely supplies a definitive latent adjacency matrix. The simulator first draws a directed block-structured truth graph at a specified sparsity level. Event histories are then generated through a stable branching Hawkes construction whose excitation respects the graph. Packet-size marks depend on latent relations and are subsequently subjected to declared obfuscation.",
            "Dense, moderate, and sparse conditions change edge prevalence while maintaining stable excitation. The same seed fixes the truth graph across architecture comparisons and obfuscation variants, which yields paired contrasts. None, padding, and jitter conditions perturb observations without changing the underlying truth. This construction isolates robustness to observation degradation rather than confounding degradation with a new graph.",
            "Simulation does not establish real-world operational performance. It establishes whether the estimator can recover known structure under a transparent generative family and controlled violations. External authorized traces then test computational feasibility and qualitative stability without pretending that their unknown graph is labeled truth.",
        ],
    ),
    (
        "Factorial Design and Randomization",
        [
            "The confirmatory design crosses two architectures, three obfuscation levels, and three sparsity levels within each of 100 independent truth seeds. Every seed appears in all 18 cells, producing 1,800 architecture runs. This is a randomized complete-block repeated-measures factorial because cell conditions are not separately assigned to higher-level whole plots.",
            "Within each seed, cell execution order is pseudorandomly permuted to reduce temporal compute drift. The seed identifies the shared latent graph and event-generation randomness. Restartable output records each completed cell, allowing interruption without double counting. The analyzer requires the complete expected cell set before producing confirmatory summaries.",
            "Figure 3 summarizes allocation. Architecture is sum-coded. Ordered three-level factors use orthogonal linear and quadratic contrasts, while interaction terms test whether architecture effects change across degradation and sparsity. The primary estimand is a paired population-average architecture contrast over the frozen seed distribution.",
        ],
    ),
    (
        "Primary and Secondary Estimands",
        [
            "Link area under the receiver operating characteristic curve is the primary discrimination endpoint on controlled directed non-diagonal pairs. It measures ranking, not probability calibration, and can become unstable when a truth graph contains too few positive or negative edges. The simulator enforces eligible graphs and reports any failed eligibility as a reliability outcome.",
            "Binary link log score is the calibration-sensitive companion. Scores are clipped only at floating-point probability boundaries, and lower negative log loss indicates better probabilistic prediction. Compute latency is measured per retained event after device synchronization, excluding acquisition wait and disk operations. Right-skewed latency is analyzed on a logarithmic scale when prespecified diagnostics warrant it.",
            "Occupied latent blocks, optimizer failure, importance effective sample size, memory use, and posterior entropy are secondary. Importance ESS belongs only to the variational architecture. Live-network outputs include event count, node count, latency, occupied blocks, and sensitivity across windows; live link AUC is unavailable without a blinded truth graph.",
        ],
    ),
    (
        "Mixed-Effects Analysis",
        [
            "For each endpoint, the fixed-effects design includes architecture, two orthogonal contrasts for obfuscation, two for sparsity, and all interactions. Seed receives a random intercept because repeated cells share a truth graph. A random architecture slope is prespecified and retained when the covariance model is estimable; singular fits trigger a documented simplified covariance structure rather than silent model switching.",
            "The architecture coefficient under sum coding estimates an average contrast across the other factors. Architecture-by-linear-obfuscation interaction measures systematic change in that contrast as degradation progresses. Quadratic interactions test nonmonotonic response. Estimated marginal means and multiplicity-adjusted simple effects are reported only after a global interaction warrants decomposition.",
            "Residual diagnostics inspect skew, leverage, heteroscedasticity, and tail behavior. Cell-specific residual scale or a parametric bootstrap is used when Gaussian constant-variance assumptions fail. Seed-clustered uncertainty provides a robustness analysis. The reporting unit is the seed-level paired contrast, not each directed dyad, thereby avoiding pseudoreplication.",
        ],
    ),
    (
        "Convergence and Approximation Diagnostics",
        [
            "Optimization diagnostics include finite objectives, gradient norms, objective stabilization, posterior entropy, and sensitivity across independent initializations. These quantities diagnose numerical behavior but do not prove posterior accuracy. Importance-weight diagnostics use normalized weights to compute a method-specific effective sample fraction; severe degeneracy indicates that the amortized approximation undercovers important regions.",
            "Rank-normalized split R-hat and autocorrelation-based effective sample size apply to genuine Markov chains. The repository includes a correct folded rank diagnostic utility for future particle-MCMC or rejuvenation extensions, but the current variational experiment does not generate those chains. Reporting R-hat from epochs would violate its sampling interpretation.",
            "Static-model optimization receives comparable finite-value and restart checks. Failure rates are reported by cell. A method that returns a result only in easy conditions cannot gain an unfair performance advantage through complete-case filtering. Figure 4 locates mathematical tests before empirical execution in the evidence ladder.",
        ],
    ),
    (
        "Latency and Streaming Constraints",
        [
            "Real-time feasibility depends on model compute per event, window width, arrival rate, memory, and acquisition overhead. The confirmatory latency endpoint isolates inference compute with CUDA synchronization where applicable. Network wait, TShark startup, and file compression are measured separately because they answer operational engineering questions rather than algorithmic throughput.",
            "The current live command performs a bounded capture followed by inference. The queue abstraction also supports rolling windows, but a production streaming scheduler would need backpressure, checkpointing, state expiration, and clock-quality controls. Late or reordered events should be handled by a declared watermark policy; silently sorting an unlimited stream introduces unbounded latency and hindsight.",
            "Hardware, software versions, thread counts, device type, and warm-up policy must accompany every latency claim. GPU acceleration is not assumed. A CPU result remains valid for its environment, while cross-device comparisons require calibration or separate strata. No deadline claim is made until execution under a declared workload.",
        ],
    ),
    (
        "Software Verification",
        [
            "The permanent test suite spans mathematical invariants and end-to-end behavior. Hawkes tests require finite likelihoods and stability-bounded excitation. Flow tests check inverse reconstruction and cancellation of forward and reverse log determinants. Simulator tests verify directed, nontrivial truth graphs and monotone event times.",
            "Privacy tests require deterministic keyed pseudonymization, distinct outputs under distinct keys, a six-field persisted schema, integrity checksum verification, and rejection of forbidden fields. Model smoke tests require finite dynamic and static outputs on the same controlled batch. Diagnostic tests distinguish importance ESS from chain diagnostics.",
            "Continuous integration installs a CPU PyTorch wheel, lints the package, runs all tests, rebuilds the report, and verifies both its page count and the dashboard claim language. Passing continuous integration demonstrates reproducible software behavior in that environment; it does not create an empirical result. Empirical status remains a separate machine-readable field.",
        ],
    ),
    (
        "Reproducibility and Provenance",
        [
            "The repository records package constraints, source code, tests, experimental design, machine-readable status, and report-generation code. Controlled simulations are seed-addressable. Factorial output is append-only and restartable, with one unique row per seed and cell. Confirmatory analysis checks the expected row count and unique key set before estimation.",
            "Private captures, salts, environment files, credentials, result directories, and packet-capture extensions are excluded through version-control rules. Public releases should contain only aggregate metrics, fixed protocol metadata, and non-sensitive figures. If live aggregates are added, their provenance must record capture duration, permitted interface class, software versions, checksum, and exclusion rules without exposing the interface or endpoint map.",
            "The PDF and dashboard are generated artifacts but remain in the repository so a reader can inspect the frozen narrative without executing a build. The source scripts are authoritative. Continuous integration verifies that the generated report has exactly 27 pages and that the dashboard preserves the non-operational claim boundary.",
        ],
    ),
    (
        "Live External Feasibility Protocol",
        [
            "After controlled hyperparameters and endpoints are frozen, the operator may acquire bounded metadata from an authorized local interface. Windows are processed in chronological order. The HMAC key remains fixed within the study to preserve relational continuity and is never published. The capture block is retained locally for only the approved duration.",
            "Because live truth is unknown, evaluation is predictive and sensitivity-based. Report prequential event log score where the observation model permits, latency per event, occupied blocks, posterior entropy, graph stability across adjacent windows, and perturbation sensitivity. Never convert internal consistency into a link-accuracy statement. A separately constructed blinded testbed would be required for externally valid live link truth.",
            "A successful live run would support the statement that the pipeline processed hardware-observed encrypted-traffic metadata under the stated privacy and compute conditions. It would not support identity, intent, protocol-content, or C2 attribution. This narrow language is preserved in both the dashboard and machine-readable status.",
        ],
    ),
    (
        "Threats to Validity",
        [
            "Construct validity is limited by the gap between pseudonymous endpoint association and meaningful application topology. Receiver artifacts, address translation, multiplexing, retransmission, and background services can create edges unrelated to an application-level structure. The model estimates the topology represented in the chosen metadata layer, not a universal network map.",
            "Internal validity depends on correct truth generation, randomized order, stable software, and complete reporting of failures. Simulation-model alignment may favor the Hawkes architecture, so misspecification scenarios and the static comparator are necessary. External validity is limited to authorized interfaces, capture windows, and obfuscation mechanisms represented in the protocol.",
            "Statistical conclusion validity depends on seed-level replication, appropriate repeated-measures covariance, endpoint multiplicity, and absence of outcome-driven protocol changes. One hundred seeds target stable paired estimates but do not guarantee power for every high-order interaction. Confidence intervals, not thresholded significance alone, carry the primary interpretation.",
        ],
    ),
    (
        "Discussion and Conclusion",
        [
            "The corrected framework changes the project from an appealing demonstration into a falsifiable statistical protocol. Exact point-process normalization prevents intensity inflation; invertible flow accounting restores a valid variational density; controlled truth supplies link labels; privacy minimization constrains the observation; and diagnostic discipline prevents optimization output from masquerading as MCMC sampling evidence.",
            "The central scientific question is whether continuous-time excitation and flexible latent geometry improve directed topology recovery when timing and size information are degraded. The design can answer that question within its simulator and compute environment. It cannot establish why a live endpoint communicates, who controls it, or whether it belongs to an adversarial structure.",
            "This division is productive rather than limiting. Controlled truth supports quantitative architecture claims, while authorized live metadata tests whether acquisition, sanitization, and inference operate on real receiver output. A later hardware-in-the-loop campaign with independently blinded truth could bridge the two lanes without retrospectively redefining success.",
            "The present conclusion is deliberately procedural: the mathematical and engineering contradictions of the baseline have been repaired, and the empirical study is ready to execute. No live traffic, 1,800-cell result, or operational validation is bundled. Therefore no performance contrast is reported. The next valid evidence is generated by running the frozen controlled experiment and a separately documented authorized live feasibility study.",
        ],
    ),
]

REFERENCES_A = [
    "Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. Biometrika, 58(1), 83–90. https://doi.org/10.1093/biomet/58.1.83",
    "Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. arXiv. https://arxiv.org/abs/1611.07308",
    "Matias, C., & Miele, V. (2017). Statistical clustering of temporal networks through a dynamic stochastic block model. Journal of the Royal Statistical Society: Series B, 79(4), 1119–1141. https://doi.org/10.1111/rssb.12200",
    "National Institute of Standards and Technology. (2024). Cybersecurity Framework 2.0. https://www.nist.gov/cyberframework",
    "Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked autoregressive flow for density estimation. arXiv. https://arxiv.org/abs/1705.07057",
    "Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. arXiv. https://arxiv.org/abs/1505.05770",
]

REFERENCES_B = [
    "Scapy Project. (n.d.). Usage. https://scapy.readthedocs.io/en/latest/usage.html",
    "The Menlo Report. (2012). Ethical principles guiding information and communication technology research. U.S. Department of Homeland Security.",
    "Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. Bayesian Analysis, 16(2), 667–718. https://doi.org/10.1214/20-BA1221",
    "Wireshark Foundation. (n.d.). TShark manual page. https://www.wireshark.org/docs/man-pages/tshark.html",
    "Internet Architecture Board. (2013). Privacy considerations for Internet protocols (RFC 6973). https://www.rfc-editor.org/rfc/rfc6973",
    "Wasserman, L. (2006). All of nonparametric statistics. Springer.",
]


def build() -> Path:
    figures = make_figures()
    pdfmetrics.registerFont(TTFont("ReportSerif", REGULAR_FONT))
    pdfmetrics.registerFont(TTFont("ReportSerifBold", BOLD_FONT))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPO_COPY.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "APA Body",
        parent=styles["BodyText"],
        fontName="ReportSerif",
        fontSize=11.5,
        leading=20.5,
        spaceAfter=8,
        alignment=TA_LEFT,
        firstLineIndent=0.5 * inch,
        textColor=colors.HexColor("#111111"),
    )
    no_indent = ParagraphStyle("No indent", parent=body, firstLineIndent=0)
    heading = ParagraphStyle(
        "APA Heading",
        parent=styles["Heading1"],
        fontName="ReportSerifBold",
        fontSize=12,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=12,
        textColor=colors.black,
    )
    title = ParagraphStyle(
        "Title",
        parent=heading,
        fontSize=17,
        leading=25,
        spaceAfter=24,
    )
    ref = ParagraphStyle(
        "Reference",
        parent=body,
        firstLineIndent=-0.5 * inch,
        leftIndent=0.5 * inch,
        leading=20,
        spaceAfter=10,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontSize=10.5,
        leading=16,
        firstLineIndent=0,
        spaceBefore=4,
        textColor=colors.HexColor("#333333"),
    )

    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("ReportSerif", 11)
        canvas.drawRightString(letter[0] - 0.75 * inch, letter[1] - 0.58 * inch, str(doc.page))
        if doc.page > 1:
            canvas.setFont("ReportSerif", 8)
            canvas.setFillColor(colors.HexColor("#555555"))
            canvas.drawString(0.75 * inch, 0.48 * inch, "ENCRYPTED PROTOCOL TOPOLOGY RECOVERY")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=0.82 * inch,
        bottomMargin=0.72 * inch,
        title="Encrypted Protocol Topology Recovery",
        author="Salem Morelli",
        subject="APA-style computational statistics methodology report",
    )
    story = []

    # Page 1: APA-style title page.
    story.extend(
        [
            Spacer(1, 1.35 * inch),
            Paragraph("Encrypted Protocol Topology Recovery From Authorized Packet Metadata", title),
            Spacer(1, 0.45 * inch),
            Paragraph("Salem Morelli", heading),
            Paragraph("Independent Computational Statistics Research", heading),
            Spacer(1, 0.7 * inch),
            Paragraph("Methodology and Preregistered Validation Report", heading),
            Paragraph("August 30, 2026", heading),
            Spacer(1, 0.75 * inch),
            Paragraph(
                "Author Note. This report documents a methodology-ready software artifact. "
                "It contains no live packet capture and no completed factorial result. "
                "Correspondence and repository provenance should be associated with the public project record.",
                no_indent,
            ),
            PageBreak(),
        ]
    )

    figure_pages = {
        5: (figures["likelihood"], "Figure 1. Exact Hawkes likelihood decomposition."),
        13: (figures["privacy"], "Figure 2. Data-minimizing acquisition path."),
        16: (figures["design"], "Figure 3. Frozen 2 × 3 × 3 repeated-measures allocation."),
        19: (figures["ladder"], "Figure 4. Evidence gates and execution boundary."),
    }
    for page_index, (page_title, paragraphs) in enumerate(PAGES, start=2):
        story.append(Paragraph(page_title, heading))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, no_indent if page_title == "Abstract" else body))
        if page_index in figure_pages:
            path, cap = figure_pages[page_index]
            story.append(Spacer(1, 6))
            story.append(Image(str(path), width=5.7 * inch, height=2.25 * inch, kind="proportional"))
            story.append(Paragraph(cap, caption))
        story.append(PageBreak())

    # Page 26: References, part 1.
    story.append(Paragraph("References", heading))
    for item in REFERENCES_A:
        story.append(Paragraph(item, ref))
    story.append(
        Paragraph(
            "References are presented in APA-style form. URLs identify the authoritative source or persistent record used to define the implemented model, diagnostics, and capture interface.",
            no_indent,
        )
    )
    story.append(PageBreak())

    # Page 27: References, part 2 and reproducibility note.
    story.append(Paragraph("References (Continued)", heading))
    for item in REFERENCES_B:
        story.append(Paragraph(item, ref))
    story.append(Spacer(1, 8))
    table = Table(
        [
            ["Artifact", "Verification"],
            ["Statistical engine", "14 permanent tests and Ruff lint"],
            ["Factorial allocation", "18 cells × 100 shared truth seeds"],
            ["Live capture", "Authorization-gated; no bundled trace"],
            ["Report", "Generated from source; exactly 27 pages"],
        ],
        colWidths=[2.0 * inch, 4.2 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "ReportSerif", 10),
                ("FONT", (0, 0), (-1, 0), "ReportSerifBold", 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eef5")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#78909c")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    pages = len(PdfReader(str(OUTPUT)).pages)
    if pages != 27:
        raise RuntimeError(f"report must contain exactly 27 pages, found {pages}")
    shutil.copy2(OUTPUT, REPO_COPY)
    return OUTPUT


if __name__ == "__main__":
    print(build())
