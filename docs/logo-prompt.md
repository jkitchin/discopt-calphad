# Logo prompt: discopt-calphad

Written to match the discopt-mkm logo (`../discopt-mkm/images/logo.png`): a
dark rounded-square app icon with glowing flat line art. The prompt is for an
image model (gpt-image, Midjourney, DALL·E, and similar). Suggested outputs
are `images/logo.png` (1024×1024), `logo-512.png`, and a wide `logo-banner.png`.

## Prompt

> A minimalist, modern app-icon logo for "discopt-calphad", a scientific
> software package that computes *certified* phase diagrams of alloys. It
> should be in the same family as a sibling icon: a rounded-square tile in deep
> midnight navy (#141d2e), soft drop shadow, off-white background, flat vector
> style, and clean glowing strokes with rounded caps. There is no text inside
> the tile.
>
> **Top half: a binary phase diagram.** Draw a stylized temperature–composition
> phase diagram as thin neon-cyan (#3ef2ff → #14b8ff) line art: a eutectic "V",
> with two liquidus curves falling from the upper corners to a eutectic point,
> a horizontal eutectic line, and a small dome-shaped miscibility gap on one
> side. Fill the two-phase regions with a very faint cyan tint. Draw one
> horizontal tie-line across a two-phase region, ending in two small glowing
> dots that mark the coexisting phase compositions.
>
> **Bottom half: the certificate.** Below the diagram, draw a double-well Gibbs
> energy curve G(x) with a warm magenta-to-orange gradient stroke (#ec4899 →
> #f59e0b). Draw a straight **common-tangent line** in the same gradient that
> touches both wells at exactly two points, marked by small hollow circles. The
> two contact points line up vertically with the tie-line endpoints above, joined
> by faint dotted guide lines. This connects the energy picture to the diagram.
> Under the tangent, draw a thin horizontal rule. Just beneath the curve, add
> a subtle band of short evenly spaced tick marks or a faint shaded strip. It
> suggests a proven lower bound, with the whole curve lying on or above the
> tangent.
>
> **Accents (small, secondary).**
> - In one corner of the diagram, a tiny checkmark inside a hexagon or seal
>   stands for "globally certified".
> - Faintly, behind the phase diagram, a small crystal-lattice motif: two
>   interpenetrating sublattices of dots in two shades of cyan. It hints at the
>   multi-sublattice compound energy formalism.
> - Optionally, a very faint ghost of symbolic math (for example a stylized
>   "Σ y ln y") at low opacity in the background texture. It must stay
>   illegible at small sizes and must not compete with the main elements.
>
> **Composition and style.**
> - Centered, balanced and generous negative space.
> - At most 6–8 strokes of meaning, so it reads clearly at 64 px.
> - No 3D, no photorealism, no gradients on the tile itself.
> - Crisp vector look, like an Apple or Material app icon.

## Negative prompt

text, letters, words, watermark, cluttered, photorealistic, 3D render, bevel,
busy grid lines, axis labels, numbers, more than two colors families, thin
illegible details, skeuomorphic

## Banner variant

> Same icon on the left. To its right, the wordmark "discopt-calphad" in a
> clean geometric sans (Inter or SF Pro, semibold). "discopt" is off-white and
> "-calphad" uses the cyan gradient. Below it, a small tagline in muted gray:
> "certified phase equilibria". The background is deep navy, in a wide 3:1
> layout.

## What each element stands for

| Element | Meaning |
|---|---|
| Eutectic diagram, tie-line, miscibility dome | phase-diagram computation and tracing |
| Double-well G(x) and common tangent | restricted equilibrium, where μ is the tangent |
| Band under the tangent, seal with checkmark | global pricing lower bound: a *certified* result |
| Two sublattices of dots | compound energy formalism (multi-sublattice phases) |
| Faint Σ y ln y | symbolic (SymPy) model representation |
| Cyan + magenta/orange on navy | visual family with discopt-mkm |
