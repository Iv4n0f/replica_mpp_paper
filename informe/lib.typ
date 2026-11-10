#let proyecto(
  title: "",
  members: (),
  teacher: "",
  logo: none,
  course: "",
  body,
) = {
  set document(title: title, author: members)
  set page(
    paper: "us-letter",
    numbering: "1",
    number-align: center,
  )
  set text(lang: "es")
  set par(justify: true)
  set page(numbering: none)
  align(center)[
    #text(size: 16pt, weight: "bold")[
      UNIVERSIDAD NACIONAL DE SAN AGUSTÍN \
      FACULTAD DE INGENIERÍA DE PRODUCCIÓN Y SERVICIOS
    ]
    #v(1fr)
    #if logo != none {
      image(logo, width: 60%)
    }
    #v(1fr)
    #text(size: 14pt, weight: "bold")[
      ESCUELA PROFESIONAL DE CIENCIA DE LA COMPUTACIÓN \
      #course
    ]
    #v(1em)
    #line(length: 100%)
    #text(size: 16pt, weight: "bold")[#title]
    #line(length: 100%)
    #v(1fr)
    #grid(
      columns: (1.2fr, 0.8fr),
      align: left,
      column-gutter: 1em,
      [
        *INTEGRANTES:* \
        #v(0.5em)
        #members.map(it => [- #it]).join()
      ],
      [
        *DOCENTE:* \
        #v(0.5em)
        #teacher
      ],
    )
    #v(1fr)
    #text(weight: "bold")[
      AREQUIPA – PERÚ \
      2026
    ]
  ]
  pagebreak()
  outline(indent: 2em)
  pagebreak()
  set page(
    numbering: "1",
    header: context {
      if counter(page).get().first() >= 1 [
        #set text(size: 8pt, fill: black)
        #grid(
          columns: (1fr, 1fr),
          align(left)[
            UNIVERSIDAD NACIONAL DE SAN AGUSTÍN DE AREQUIPA \
            ESCUELA PROFESIONAL DE CIENCIA DE LA COMPUTACIÓN
          ],
          align(right)[
            CURSO: #course
          ],
        )
        #line(length: 100%, stroke: 0.5pt + black)
      ]
    },
  )
  counter(page).update(1)
  set heading(numbering: "1.")

  // ---------- Heading styles ----------
  show heading.where(level: 1): it => {
    v(1em)
    set text(size: 15pt, weight: "bold", fill: black)
    block(it)
    v(0.2em)
  }
  show heading.where(level: 2): it => {
    v(0.8em)
    set text(size: 12.5pt, weight: "bold", fill: black)
    block(it)
  }
  show heading.where(level: 3): it => {
    v(0.6em)
    set text(size: 11.5pt, weight: "bold", style: "italic")
    block(it)
  }

  // ---------- Inline code style for technical terms ----------
  show raw.where(block: false): it => box(
    fill: rgb("#f2f2f2"),
    outset: (x: 2pt, y: 2pt),
    radius: 2pt,
    text(font: "DejaVu Sans Mono", size: 9.6pt, fill: black, it),
  )

  set list(marker: [•], indent: 0.4em)
  set enum(indent: 0.4em)

  body
}

// ---------- Reference citation marker (superscript [n]) ----------
#let footnote_ref(n) = super(text(size: 8.5pt, fill: black)[[#n]])

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1"
)

#set text(
  font: ("Linux Libertine O", "Liberation Serif", "DejaVu Serif"),
  size: 11pt,
  lang: "es"
)

#set par(
  justify: true,
  leading: 0.65em
)