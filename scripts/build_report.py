"""Build the exact 27-page simulation-only statistical report."""

from __future__ import annotations

import argparse
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
FIGURES = ROOT / "output" / "figures"
REPO_FIGURES = ROOT / "report" / "figures"
FONT_PATHS = (
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    ),
    (Path("C:/Windows/Fonts/times.ttf"), Path("C:/Windows/Fonts/timesbd.ttf")),
    (
        Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
    ),
)
RESULTS_SHA256 = "d19433ff32f2222655f5408810b9ed6fd59fa01ea461e05e3f1aa5a59977309a"


def register_report_fonts() -> tuple[str, str]:
    """Use a platform font when available and ReportLab's built-ins otherwise."""

    for regular_path, bold_path in FONT_PATHS:
        if regular_path.exists() and bold_path.exists():
            pdfmetrics.registerFont(TTFont("ReportSerif", regular_path))
            pdfmetrics.registerFont(TTFont("ReportSerifBold", bold_path))
            return "ReportSerif", "ReportSerifBold"
    return "Times-Roman", "Times-Bold"


def make_figures() -> dict[str, Path]:
    """Generate only model, design, and simulation schematics."""

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    names = ["Event log-intensity", "Baseline compensator", "Excitation compensator"]
    values = [1, -1, -1]
    bars = ax.barh(names, values, color=["#21b6d7", "#ffbe55", "#ff7b73"])
    ax.axvline(0, color="#243b4a", lw=1)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_title("All three terms define the point-process likelihood", weight="bold")
    for bar, label in zip(bars, ["Σ log λ", "−TΣμ", "−ΣA(1−e⁻ᵝᵘ)"], strict=True):
        ax.text(
            bar.get_width() * 0.52,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="center",
            va="center",
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    likelihood = FIGURES / "likelihood_decomposition.png"
    fig.savefig(likelihood, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].imshow(np.arange(9).reshape(3, 3), cmap="Blues", alpha=0.85)
    axes[0].set_xticks(range(3), ["Dense", "Moderate", "Sparse"])
    axes[0].set_yticks(range(3), ["None", "Padding", "Jitter"])
    axes[0].set_xlabel("Graph sparsity")
    axes[0].set_ylabel("Perturbation")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, "2 models", ha="center", va="center", fontsize=8)
    axes[0].set_title("18 cells per seed and generator", weight="bold")
    axes[1].bar(
        ["Corrected\npending", "Robustness\npending"],
        [1800, 2700],
        color=["#d9dfe4", "#d9dfe4"],
    )
    axes[1].set_ylabel("Architecture runs")
    axes[1].set_title("Separate result schemas", weight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    design = FIGURES / "factorial_design.png"
    fig.savefig(design, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    labels = [
        "Truth\ngraph",
        "Registered\nevent law",
        "Synthetic\nperturbation",
        "Two-model\ncomparison",
        "Paired\ninference",
    ]
    xs = np.linspace(0.9, 9.1, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels, strict=True)):
        color = "#e9e0ff" if index == 1 else "#d9f6fb"
        ax.add_patch(
            plt.Rectangle((x - 0.72, 0.78), 1.44, 0.86, fc=color, ec="#28738b", lw=1.3)
        )
        ax.text(x, 1.21, label, ha="center", va="center", weight="bold", fontsize=8)
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(xs[index + 1] - 0.78, 1.21),
                xytext=(x + 0.78, 1.21),
                arrowprops={"arrowstyle": "->", "color": "#28738b"},
            )
    ax.text(
        5,
        0.3,
        "All events, node labels, marks, graphs, and truth labels are generated in memory",
        ha="center",
        color="#274a59",
        weight="bold",
    )
    ax.set_title("Simulation-only evidence path", weight="bold")
    fig.tight_layout()
    registry = FIGURES / "simulation_registry.png"
    fig.savefig(registry, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    labels = [
        "Likelihood\ntests",
        "Flow and\nmodel tests",
        "Corrected\nfactorial",
        "Robustness\nsuite",
        "Independent\nreplication",
    ]
    completed = [True, True, False, False, False]
    ax.bar(
        range(5),
        [1] * 5,
        color=["#51d49d" if item else "#d9dfe4" for item in completed],
        edgecolor="#426071",
    )
    ax.set_xticks(range(5), labels)
    ax.set_yticks([])
    ax.set_ylim(0, 1.25)
    ax.set_title("Evidence gates preserve completed and pending status", weight="bold")
    for index, done in enumerate(completed):
        ax.text(
            index,
            0.5,
            "complete" if done else "pending",
            ha="center",
            va="center",
            weight="bold",
            fontsize=8,
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    ladder = FIGURES / "evidence_ladder.png"
    fig.savefig(ladder, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"likelihood": likelihood, "design": design, "registry": registry, "ladder": ladder}


PAGES: list[tuple[str, list[str]]] = [
    (
        "Abstract",
        [
            "This report documents a simulation-only computational statistics laboratory for recovering latent directed topology under controlled truth. Version 1.2.0 contains no collection, interface-enumeration, external-trace import, address mapping, or live-inference path. Every event, mark, node label, graph, and truth label used by the runtime is generated in memory from a declared integer seed.",
            "The fitted architecture combines an exact exponential-kernel multivariate Hawkes likelihood, an amortized dynamic stochastic block model, and masked autoregressive variational flows. A static graph autoencoder with Louvain clustering provides the comparator. The likelihood includes both event log-intensities and the exact observation-window compensator; the variational density includes the flow Jacobian; and excitation is constrained below a stability boundary.",
            "A prior 1,800-row summary is retained only as historical, hash-bound material. Audit v1.2.0 found that parameter initialization preceded the declared cell seed, the source CSV was absent, and the simulated observation-window convention required correction. Those values are quarantined and support no current architecture ranking. A corrected 1,800-cell rerun and the separate five-generator, 2,700-cell robustness study are pending.",
            "Keywords: Hawkes process, stochastic block model, normalizing flow, simulation, topology recovery, model misspecification, negative control",
        ],
    ),
    (
        "Research Question and Scope",
        [
            "The scientific question is whether continuous-time excitation and flexible latent geometry improve directed graph recovery when synthetic timing and size information are degraded. The estimand is a difference between two algorithms evaluated on the same generated truth graph, not a statement about any real entity or communication system.",
            "Controlled truth separates recovery error from interpretive uncertainty. A simulator supplies a directed binary adjacency matrix and event observations whose stochastic relationship to that matrix is known. Because every design coordinate is explicit, failures can be attributed to architecture, perturbation, sparsity, generator mismatch, or sampling variation rather than an undocumented source.",
            "The design cannot decrypt content, identify a person or organization, infer intent, attribute command and control, or establish operational SIGINT performance. Those quantities are absent from the data-generating process and therefore unidentified by the model. This boundary is enforced in source layout, command-line surface, dependencies, tests, status metadata, and publication artifacts.",
        ],
    ),
    (
        "Synthetic Observation Space",
        [
            "Events live on the product space of continuous time, directed synthetic dyads, and positive synthetic size marks. The counting measure records occurrences on this space, and the predictable intensity is measurable with respect to the left-continuous event history. Node keys are labels such as synthetic-node-003 and have no external correspondence.",
            "The event batch contains aligned tensors for times, sizes, sources, and destinations. Validation requires nondecreasing nonnegative times, positive sizes, in-range node indices, and no self-links. These conditions define an auditable mathematical object independent of any external serialization or collection system.",
            "The finite observation window is the declared interval from zero through the registered horizon; the initial event-free exposure and terminal exposure are retained. New event laws are selected through an explicit registry field and write to a separate result schema, preventing a robustness extension from silently redefining the corrected primary experiment.",
        ],
    ),
    (
        "Exact Multivariate Hawkes Likelihood",
        [
            "For event type r, conditional intensity is a positive baseline plus exponentially decayed contributions from earlier types. The log likelihood on a finite interval equals the sum of observed event log-intensities minus integrated intensity over the full event-type domain. This compensator normalizes the process and cannot be omitted.",
            "With an exponential kernel, the baseline contribution is window length times the sum of baselines. Each prior event contributes its excitation coefficient multiplied by one minus an exponential tail factor. The implementation uses positive smooth transforms, a declared numerical floor, and finite-gradient checks.",
            "Figure 1 shows the sign and role of the three components. The event term rewards intensity at observed events, while both compensator components penalize intensity integrated over exposure. A model that retains only the first term can increase its objective by inflating intensity and does not represent the intended probability law.",
        ],
    ),
    (
        "Stability and Partial Identifiability",
        [
            "A stationary linear Hawkes process requires excitation below its critical branching boundary. The software rescales the nonnegative excitation matrix to a spectral-norm ceiling of 0.85. This conservative differentiable constraint keeps the fitted process inside the implemented stability region.",
            "Baseline and excitation are only partially identifiable in short sequences. A high baseline with weak excitation can resemble a lower baseline with stronger history dependence. Block labels also switch without altering likelihood. Evaluation therefore uses permutation-invariant link scores and never attaches semantic meaning to a block number.",
            "The robustness registry makes these limitations testable. A mixture kernel probes time-scale mismatch, a Cox law introduces shared rate changes, a renewal law creates non-Poisson intervals without self-excitation, and a null law removes topology information. Agreement under one generator is not treated as universal identification.",
        ],
    ),
    (
        "Synthetic Marks and Perturbations",
        [
            "Synthetic size is modeled as a conditional mark associated with the generated block relation. The primary event-time likelihood remains a multivariate counting process. This factorization declares that, conditional on latent structure and dyadic type, the size contribution does not alter the event-time compensator.",
            "The padding condition maps all observed synthetic sizes to 1,500 units and injects declared dummy events. The jitter condition adds seeded Gaussian displacement to generated times, clips to the window, and stably reorders observations. The none condition retains the generated values. Truth adjacency remains unchanged across perturbations.",
            "These perturbations are abstractions rather than copies of a field system. They test loss or contamination of model inputs within a controlled experiment. Generalization is limited to the coded mechanisms and parameter values, so any stronger robustness statement requires additional registered laws and a new frozen design.",
        ],
    ),
    (
        "Dynamic Stochastic Block Structure",
        [
            "The block model represents each synthetic node by a probability vector over a finite truncation of latent roles. Inferred block-specific baseline event rates generate asymmetric link scores. Relaxed Gumbel-Softmax draws make discrete allocations differentiable during optimization while allowing unused blocks to receive negligible mass.",
            "Node features summarize generated outgoing and incoming counts, size moments, and timing rates. An amortized encoder maps these features to variational parameters. The resulting embeddings enter both relaxed block allocation and the directed link decoder.",
            "Blocks are structural abstractions, not identities or categories. Their labels can exchange under posterior symmetry. The defensible object is a posterior over synthetic relational structure and occupied components, evaluated against the simulator's binary adjacency matrix.",
        ],
    ),
    (
        "Masked Autoregressive Variational Flow",
        [
            "A diagonal Gaussian posterior may be too rigid for graph-allocation uncertainty. A masked autoregressive flow transforms Gaussian base draws through triangular bijections. Each coordinate shift and scale depends only on preceding coordinates under the selected ordering, yielding a tractable Jacobian.",
            "Bounded log scales prevent numerical explosion while preserving invertibility. Alternating coordinate order improves dependence coverage. Forward and inverse tests require reconstruction within numerical tolerance and cancellation of the corresponding log determinants.",
            "The flow increases approximation flexibility but does not make the posterior exact. Importance effective sample fraction describes degeneracy of variational importance weights only. It is not Markov-chain effective sample size and is not used as a cross-architecture endpoint.",
        ],
    ),
    (
        "Evidence Lower Bound and Optimization",
        [
            "The evidence lower bound is the Monte Carlo expectation of joint log density minus variational log density. The joint term combines the exact event-time likelihood, synthetic mark contribution, block-structured graph prior, and regularization. The variational term includes Gaussian base density and every flow log Jacobian.",
            "Optimization uses Adam, gradient clipping, seeded initialization, and finite-value guards. Device synchronization surrounds measured compute segments when acceleration is available. The final objective per event is a training diagnostic rather than a confirmatory architecture score because the two methods optimize different objectives.",
            "Variational restarts and epochs are not Markov chains. The project therefore does not compute Gelman-Rubin statistics from optimization histories. Rank-normalized folded R-hat remains appropriate only for future algorithms that produce genuine independent chains.",
        ],
    ),
    (
        "Static Comparator",
        [
            "The comparator aggregates directed synthetic event counts into a static graph, encodes node features with a graph autoencoder, and decodes link probabilities from the latent representation. Louvain partitioning supplies an occupied-community count for descriptive comparison.",
            "It deliberately lacks continuous-time intensity and cannot represent event-history excitation. That limitation defines the architecture contrast: whether the additional temporal likelihood and flexible posterior improve link recovery enough to justify their computational cost.",
            "Both methods receive the same generated batch and truth labels within a seed and condition. Identical scoring code computes off-diagonal link AUC and binary link log score. Latency is normalized by event count; architecture-specific diagnostics remain separate.",
        ],
    ),
    (
        "Simulation-Only Software Boundary",
        [
            "The executable interface contains only simulation-registry, simulate, crypto-lab, run-factorial, analyze, run-robustness, and analyze-robustness. There is no interface enumeration, acquisition backend, external event-file argument, address transformer, or live-inference command.",
            "The package exposes EventBatch but no persistent external-event record. Packet-oriented dependencies are absent. Boundary tests assert that the former acquisition, storage, and pseudonymization modules cannot be imported and that the command set contains exactly the seven approved operations.",
            "This boundary is a scientific design choice. It eliminates ambiguous provenance and confines all link-truth statements to known simulation. Suitable extensions include new stochastic generators, priors, perturbations, calibration procedures, posterior-predictive checks, and compute experiments.",
        ],
    ),
    (
        "Registered Evidence Path",
        [
            "Figure 2 summarizes the only runtime evidence path. A seed produces a truth graph and block memberships. One registered law produces an event batch. A declared perturbation changes observations without changing truth. Both architectures receive that batch, and inference operates on paired seed differences.",
            "Registry metadata records name, temporal process, presence or absence of topology signal, fitted-model alignment, and scientific purpose. This metadata is emitted through the CLI and embedded in robustness summaries so a numerical result remains attached to its data-generating assumptions.",
            "All five generators return the same validated object shape. This common contract enables matched scoring while preventing informal substitution of an undocumented data source. Unknown generator names fail before random generation begins.",
        ],
    ),
    (
        "Matched Exponential Hawkes Generator",
        [
            "The primary generator draws balanced node memberships and a directed block-connectivity matrix conditional on at least one absent off-diagonal block edge. This condition guarantees both classes for link AUC without editing a degenerate draw after the fact.",
            "Block connectivity controls immigrant rates, while diagonal and row-or-column-related excitation terms create event dependence. A branching construction draws immigrant events and recursive offspring under an exponential delay law. Synthetic node pairs and sizes are then sampled within the event's source and destination blocks.",
            "Version 1.2.0 retains this generator as the default but corrects finite-window offspring sampling: total branching mass is sampled once and exponential delays are then censored once. Explicit and implicit generator selection remain identical. Because this correction and deterministic pre-construction seeding change the valid evidence path, the registered primary study must be rerun.",
        ],
    ),
    (
        "Misspecification Generators",
        [
            "The mixture-Hawkes law uses short- and long-scale offspring delays. It preserves excitation and topology signal but lies outside the fitted single exponential kernel. Performance loss isolates sensitivity to kernel-shape mismatch.",
            "The piecewise-Cox law applies shared lognormal multipliers across six time segments. It preserves topology-linked baseline differences while adding external rate shocks that may resemble event dependence. The gamma-renewal law generates independent mark-specific renewal sequences with shape below one, creating bursty intervals without self-excitation.",
            "These families do not span every possible event process. They form named, reproducible probes of three failure modes: kernel scale, nonstationarity, and dependence mechanism. Conclusions must name the generator rather than use an unrestricted claim of robustness.",
        ],
    ),
    (
        "Independent No-Signal Negative Control",
        [
            "The independent-null family draws homogeneous Poisson events with equalized mark rates unrelated to block connectivity. A binary truth graph is still drawn so the same evaluation functions run, but observations contain no recoverable topology signal by construction.",
            "Its purpose is calibration. An architecture that systematically reports strong discrimination under this law may exploit finite-sample artifacts, scoring leakage, optimization bias, or an unintended relationship in the simulator. Near-chance discrimination is the desired behavior.",
            "The null does not prove absence of a real relationship and cannot validate a decision about any entity. It is solely an internal negative control for the synthetic pipeline. Its registry record explicitly declares truth_signal false.",
        ],
    ),
    (
        "Registered Corrected Primary Factorial",
        [
            "The primary design crosses two architectures, three perturbations, and three sparsity levels within each of 100 seeds, for 18 cells per seed and 1,800 runs. Each seed is a complete block because it supplies every treatment combination.",
            "Within-block order is deterministically shuffled. Shared seeds produce paired architecture contrasts and reduce between-graph variation. Primary endpoints are link AUC, binary link log score, and compute latency per event. Failures remain part of the design status rather than being silently discarded.",
            "Figure 3 shows the nine perturbation-by-sparsity conditions and the two distinct experiment sizes. The robustness extension uses its own generator-prefixed schema; it cannot append to the corrected primary table.",
        ],
    ),
    (
        "Historical Primary Evidence — Quarantined",
        [
            "A historical summary reports 1,800 rows and identifies them with SHA-256 d19433ff32f2222655f5408810b9ed6fd59fa01ea461e05e3f1aa5a59977309a. The underlying CSV is not present in this repository, so the hash and exact design cannot be independently recomputed from the published checkout.",
            "Audit v1.2.0 reproduced a deterministic-seeding defect: dynamic-model parameters were constructed before the declared cell seed was applied. It also corrected the observation-window convention and finite-window Hawkes offspring simulation. Those changes affect the intended estimand and generated data path.",
            "Accordingly, all historical numerical contrasts are quarantined. They are retained only for traceability and support no present architecture ranking, performance conclusion, or completed-evidence claim. Corrected primary evidence requires a fresh 1,800-cell source CSV that passes the exact analyzer contract.",
        ],
    ),
    (
        "Analyzer v1.2.0 Inference",
        [
            "The analyzer computes the result CSV SHA-256 before and after reading and aborts if the file changes. It requires the exact registered seed-by-architecture-by-perturbation-by-sparsity key set, strict field order, valid numeric ranges, and complete status before publishing a summary. Every condition-specific architecture contrast is calculated from sorted seed-level paired differences.",
            "For each endpoint, nine paired two-sided tests receive Holm step-down adjustment controlling familywise error at 0.05. Confidence intervals are condition-specific unadjusted 95% intervals. The summary records raw and adjusted p values, decision flags, standard errors, and paired seed counts.",
            "The factorial model first attempts a seed random intercept plus architecture slope, then a random intercept only. Inadmissible mixed fits fall back to the same fixed design with seed-clustered OLS inference. Convergence warnings, covariance validity, and every rejection reason are retained rather than hidden.",
        ],
    ),
    (
        "Robustness Experiment",
        [
            "The robustness design crosses five generators with two architectures, three perturbations, three sparsity levels, and 30 default seeds. This yields 2,700 cells. Output is restartable and keyed by seed, generator, architecture, perturbation, and sparsity.",
            "Analysis requires the exact expected key set and complete status. Within every generator and condition, architecture differences are paired by seed. Each endpoint has 45 generator-by-condition comparisons, which receive one Holm correction family at alpha 0.05.",
            "The analyzer produces 135 contrasts across three endpoints, preserves the verified input hash, and embeds selected registry records in its JSON summary. Because the experiment is pending, this report gives no cross-generator numerical performance claim.",
        ],
    ),
    (
        "Convergence and Calibration Diagnostics",
        [
            "Optimization checks include finite objectives, finite gradients, gradient clipping, occupied components, and importance-weight effective sample fraction. These diagnose numerical behavior but do not prove posterior correctness.",
            "Generator-specific posterior-predictive checks should compare event counts, interarrival distributions, size quantiles, and synthetic dyad counts. Simulation-based calibration can be added when the full generative prior and posterior targets are aligned. The independent null directly audits spurious link discrimination.",
            "Static and dynamic failures are counted by condition. Complete-case filtering is prohibited because selective success could bias model comparison. Figure 4 distinguishes completed mathematical checks from pending corrected primary evidence, robustness, and independent replication.",
        ],
    ),
    (
        "Compute and Latency",
        [
            "The confirmatory latency endpoint measures fitting compute and divides by generated event count. Device synchronization is used where applicable. Event generation and result-file operations are excluded because the estimand is architecture compute, not end-to-end system throughput.",
            "Hardware, software versions, device type, thread configuration, and warm-up policy should accompany every latency result. Cross-device comparisons require separate strata or calibration. Acceleration is optional and a CPU result remains valid for its declared environment.",
            "No current latency contrast is claimed. The corrected primary and robustness designs will ask whether compute cost changes by architecture, event law, and perturbation. They do not make a real-time, operational, or streaming-performance claim.",
        ],
    ),
    (
        "Software Verification and Provenance",
        [
            "The permanent suite spans Hawkes likelihood and stability, flow inversion, model fitting, simulator determinism, factorial inference, generator registry behavior, exact robustness locks, and the simulation-only boundary. Continuous integration installs a CPU PyTorch wheel, lints, tests, rebuilds this report, and verifies publication claims.",
            "Experiment output is restartable and written atomically. Primary and robustness schemas are distinct. Public summaries bind aggregate inference to exact source hashes. The machine-readable status distinguishes quarantined historical metadata from the pending corrected 1,800-cell factorial and pending 2,700-cell robustness suite.",
            "Generated PDF and dashboard artifacts remain in the repository for inspection, while source scripts are authoritative. An exact 27-page check prevents accidental report drift. Passing software verification demonstrates implementation consistency, not new empirical evidence.",
        ],
    ),
    (
        "Threats to Validity",
        [
            "Internal validity depends on correct truth generation, deterministic initialization, stable dependencies, and complete failure reporting. The primary generator is close to the fitted model, which can favor the dynamic architecture in principle. No current architecture ranking is asserted before the corrected rerun.",
            "Construct validity is limited to binary synthetic block connectivity and the implemented event laws. Link AUC and log score quantify recovery of that adjacency, not semantic relationships. Padding and jitter are narrow perturbations and should not be treated as exhaustive degradation mechanisms.",
            "Statistical conclusion validity depends on seed pairing, multiplicity control, admissible covariance inference, and no outcome-driven changes. External validity is intentionally restricted to the registry. Independent replication and additional preregistered generators are needed before broad algorithmic claims.",
        ],
    ),
    (
        "Discussion and Conclusion",
        [
            "The project now provides a falsifiable simulation laboratory with an exact point-process target, invertible variational density, controlled graph truth, deterministic generator registry, static comparator, provenance-locked inference, and explicit negative control.",
            "The audit shows why reproducibility controls are part of the scientific result: a hash-bound aggregate is insufficient when the source rows are absent and the declared seed does not cover parameter initialization. The corrected primary and robustness studies are pending, so no architecture ranking is currently defensible.",
            "The strongest defensible conclusion remains narrow. This software evaluates synthetic topology recovery under controlled stochastic assumptions. It does not ingest real observations and supports no conclusion about encrypted content, identity, intent, attribution, command and control, or operational SIGINT performance.",
        ],
    ),
]

REFERENCES_A = [
    "Cox, D. R. (1955). Some statistical methods connected with series of events. Journal of the Royal Statistical Society: Series B, 17(2), 129–164.",
    "Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. Biometrika, 58(1), 83–90. https://doi.org/10.1093/biomet/58.1.83",
    "Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65–70.",
    "Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. arXiv. https://arxiv.org/abs/1611.07308",
    "Matias, C., & Miele, V. (2017). Statistical clustering of temporal networks through a dynamic stochastic block model. Journal of the Royal Statistical Society: Series B, 79(4), 1119–1141. https://doi.org/10.1111/rssb.12200",
]

REFERENCES_B = [
    "Ogata, Y. (1981). On Lewis' simulation method for point processes. IEEE Transactions on Information Theory, 27(1), 23–31. https://doi.org/10.1109/TIT.1981.1056305",
    "Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked autoregressive flow for density estimation. arXiv. https://arxiv.org/abs/1705.07057",
    "Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. arXiv. https://arxiv.org/abs/1505.05770",
    "Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization. Bayesian Analysis, 16(2), 667–718. https://doi.org/10.1214/20-BA1221",
    "Wasserman, L. (2006). All of nonparametric statistics. Springer.",
]


def build(*, publish: bool = False) -> Path:
    figures = make_figures()
    regular_font, bold_font = register_report_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if publish:
        REPO_COPY.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "APA Body",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=10.8,
        leading=18.2,
        spaceAfter=8,
        alignment=TA_LEFT,
        firstLineIndent=0.5 * inch,
        textColor=colors.HexColor("#111111"),
    )
    no_indent = ParagraphStyle("No indent", parent=body, firstLineIndent=0)
    heading = ParagraphStyle(
        "APA Heading",
        parent=styles["Heading1"],
        fontName=bold_font,
        fontSize=12,
        leading=21,
        alignment=TA_CENTER,
        spaceAfter=11,
        textColor=colors.black,
    )
    title = ParagraphStyle("Title", parent=heading, fontSize=17, leading=25, spaceAfter=24)
    reference = ParagraphStyle(
        "Reference",
        parent=body,
        firstLineIndent=-0.5 * inch,
        leftIndent=0.5 * inch,
        leading=19,
        spaceAfter=10,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontSize=10,
        leading=15,
        firstLineIndent=0,
        spaceBefore=4,
        textColor=colors.HexColor("#333333"),
    )

    def header_footer(canvas, document):
        canvas.saveState()
        canvas.setFont(regular_font, 10)
        canvas.drawRightString(
            letter[0] - 0.75 * inch, letter[1] - 0.58 * inch, str(document.page)
        )
        if document.page > 1:
            canvas.setFont(regular_font, 8)
            canvas.setFillColor(colors.HexColor("#555555"))
            canvas.drawString(0.75 * inch, 0.48 * inch, "SIMULATION-ONLY TOPOLOGY RECOVERY")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=0.82 * inch,
        bottomMargin=0.72 * inch,
        title="Simulation-Only Encrypted Protocol Topology Recovery",
        author="Salem Morelli",
        subject="Computational statistics report",
    )
    story = [
        Spacer(1, 1.25 * inch),
        Paragraph("Encrypted Protocol Topology Recovery", title),
        Paragraph("A Simulation-Only Computational Statistics Laboratory", title),
        Spacer(1, 0.35 * inch),
        Paragraph("Salem Morelli", heading),
        Paragraph("Independent Computational Statistics Research", heading),
        Spacer(1, 0.55 * inch),
        Paragraph("Version 1.2.0 · September 21, 2026", heading),
        Spacer(1, 0.55 * inch),
        Paragraph(
            "Author Note. This report covers controlled synthetic methods only. Historical "
            "1,800-row aggregates are quarantined after a reproducibility audit; both the "
            "corrected primary rerun and separate robustness experiment are pending. No "
            "external observations enter the software.",
            no_indent,
        ),
        PageBreak(),
    ]
    figure_pages = {
        5: (figures["likelihood"], "Figure 1. Exact Hawkes likelihood decomposition."),
        13: (figures["registry"], "Figure 2. Simulation-only evidence path."),
        17: (figures["design"], "Figure 3. Primary and robustness allocations."),
        23: (figures["ladder"], "Figure 4. Completed and pending evidence gates."),
    }
    for page_index, (page_title, paragraphs) in enumerate(PAGES, start=2):
        story.append(Paragraph(page_title, heading))
        for paragraph in paragraphs:
            style = no_indent if page_title == "Abstract" else body
            story.append(Paragraph(paragraph, style))
        if page_index in figure_pages:
            path, caption_text = figure_pages[page_index]
            story.append(Spacer(1, 5))
            story.append(Image(str(path), width=5.6 * inch, height=2.15 * inch, kind="proportional"))
            story.append(Paragraph(caption_text, caption))
        story.append(PageBreak())

    story.append(Paragraph("References", heading))
    for item in REFERENCES_A:
        story.append(Paragraph(item, reference))
    story.append(
        Paragraph(
            "References define the point-process, block-model, flow, graph-comparison, and "
            "multiplicity methods implemented in the repository.",
            no_indent,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("References (Continued)", heading))
    for item in REFERENCES_B:
        story.append(Paragraph(item, reference))
    story.append(Spacer(1, 8))
    hash_cell = Paragraph(
        f"{RESULTS_SHA256[:32]}<br/>{RESULTS_SHA256[32:]}",
        ParagraphStyle(
            "Table hash",
            parent=caption,
            fontName=regular_font,
            fontSize=7.5,
            leading=9,
            spaceBefore=0,
        ),
    )
    table = Table(
        [
            ["Artifact", "Verification"],
            ["Runtime boundary", "Seven simulation, crypto-lab, and analysis commands only"],
            ["Corrected factorial", "1,800 registered cells pending; analyzer v1.2.0"],
            ["Historical hash", hash_cell],
            ["Robustness suite", "Five generators; 2,700 default cells pending"],
            ["Report", "Generated from source; exactly 27 pages"],
        ],
        colWidths=[1.55 * inch, 4.65 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), regular_font, 8.7),
                ("FONT", (0, 0), (-1, 0), bold_font, 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eef5")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#78909c")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)

    document.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    pages = len(PdfReader(str(OUTPUT)).pages)
    if pages != 27:
        raise RuntimeError(f"report must contain exactly 27 pages, found {pages}")
    if publish:
        shutil.copy2(OUTPUT, REPO_COPY)
        REPO_FIGURES.mkdir(parents=True, exist_ok=True)
        for figure in figures.values():
            shutil.copy2(figure, REPO_FIGURES / figure.name)
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish",
        action="store_true",
        help="copy the verified PDF and figures into tracked publication paths",
    )
    print(build(publish=parser.parse_args().publish))
