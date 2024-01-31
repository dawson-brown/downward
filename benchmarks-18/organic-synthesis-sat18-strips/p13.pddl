(define (problem initialbonds17)
(:domain organic-synthesis)
(:objects
    ; setup for problem 17
    c1 - carbon
    c2 - carbon
    c3 - carbon
    c4 - carbon
    c5 - carbon
    c6 - carbon
    o1 - oxygen
    h1 - hydrogen
    h2 - hydrogen
    h3 - hydrogen
    h4 - hydrogen
    h5 - hydrogen
    h6 - hydrogen
    h7 - hydrogen
    h8 - hydrogen
    h27 - hydrogen
    h28 - hydrogen
    h29 - hydrogen
    h30 - hydrogen
    ; sodium hydroxide Na-OH
    na - sodium
    o50 - oxygen
    h50 - hydrogen
    ; water H-OH
    o51 - oxygen
    h51 - hydrogen
    h52 - hydrogen
    ; second starting material
    c7 - carbon
    c8 - carbon
    c9 - carbon
    o2 - oxygen
    h31 - hydrogen
    h32 - hydrogen
    h33 - hydrogen
    h34 - hydrogen
    h35 - hydrogen
    h36 - hydrogen
    h37 - hydrogen
    h38 - hydrogen
    ; third starting material
    c10 - carbon
    c11 - carbon
    c27 - carbon
    o3 - oxygen
    h39 - hydrogen
    h40 - hydrogen
    h41 - hydrogen
    h42 - hydrogen
    h43 - hydrogen
    h44 - hydrogen
    h45 - hydrogen
    h46 - hydrogen
    ; first PCC
    c12 - carbon
    c13 - carbon
    c14 - carbon
    c15 - carbon
    c16 - carbon
    n4 - nitrogen
    h9 - hydrogen
    h10 - hydrogen
    h11 - hydrogen
    h12 - hydrogen
    h13 - hydrogen
    h14 - hydrogen
    cr1 - chromium
    o4 - oxygen
    o5 - oxygen
    o6 - oxygen
    cl1 - chlorine
    ; second PCC
    c17 - carbon
    c18 - carbon
    c19 - carbon
    c20 - carbon
    c21 - carbon
    n5 - nitrogen
    h15 - hydrogen
    h16 - hydrogen
    h17 - hydrogen
    h18 - hydrogen
    h19 - hydrogen
    h20 - hydrogen
    cr2 - chromium
    o7 - oxygen
    o8 - oxygen
    o9 - oxygen
    cl2 - chlorine
    ; third PCC
    c22 - carbon
    c23 - carbon
    c24 - carbon
    c25 - carbon
    c26 - carbon
    n6 - nitrogen
    h21 - hydrogen
    h22 - hydrogen
    h23 - hydrogen
    h24 - hydrogen
    h25 - hydrogen
    h26 - hydrogen
    cr3 - chromium
    o10 - oxygen
    o11 - oxygen
    o12 - oxygen
    cl3 - chlorine
)
(:init
    ; setup for problem 17
    (bond c1 c2)
    (bond c2 c3)
    (bond c3 c4)
    (bond c4 c5)
    (bond c5 c6)
    (bond c6 c1)
    (bond c2 c1)
    (bond c3 c2)
    (bond c4 c3)
    (bond c5 c4)
    (bond c6 c5)
    (bond c1 c6)
    (bond c1 o1)
    (bond o1 c1)
    (bond c1 h1)
    (bond h1 c1)
    (bond c2 h2)
    (bond c2 h3)
    (bond c3 h4)
    (bond c3 h5)
    (bond c4 h6)
    (bond c4 h7)
    (bond c5 h8)
    (bond c5 h27)
    (bond c6 h28)
    (bond c6 h29)
    (bond h2 c2)
    (bond h3 c2)
    (bond h4 c3)
    (bond h5 c3)
    (bond h6 c4)
    (bond h7 c4)
    (bond h8 c5)
    (bond h27 c5)
    (bond h28 c6)
    (bond h29 c6)
    (bond o1 h30)
    (bond h30 o1)
    ; second starting material
    (bond c7 c8)
    (bond c8 c9)
    (bond c8 c7)
    (bond c9 c8)
    (bond c8 o2)
    (bond o2 c8)
    (bond c7 h31)
    (bond c7 h32)
    (bond c7 h33)
    (bond h31 c7)
    (bond h32 c7)
    (bond h33 c7)
    (bond c9 h34)
    (bond c9 h35)
    (bond c9 h36)
    (bond h34 c9)
    (bond h35 c9)
    (bond h36 c9)
    (bond c8 h37)
    (bond h37 c8)
    (bond o2 h38)
    (bond h38 o2)
    ; third starting material
    (bond c10 c11)
    (bond c11 c27)
    (bond c11 c10)
    (bond c27 c11)
    (bond c27 o3)
    (bond o3 c27)
    (bond c10 h39)
    (bond c10 h40)
    (bond c10 h41)
    (bond h39 c10)
    (bond h40 c10)
    (bond h41 c10)
    (bond c11 h42)
    (bond c11 h43)
    (bond h42 c11)
    (bond h43 c11)
    (bond c27 h44)
    (bond c27 h45)
    (bond h44 c27)
    (bond h45 c27)
    (bond o3 h46)
    (bond h46 o3)
    ; first PCC
    (bond n4 h9)
    (bond h9 n4)
    (AROMATICBOND c12 n4)
    (AROMATICBOND c12 c13)
    (AROMATICBOND c13 c14)
    (AROMATICBOND c14 c15)
    (AROMATICBOND c15 c16)
    (AROMATICBOND c16 n4)
    (AROMATICBOND n4 c12)
    (AROMATICBOND c13 c12)
    (AROMATICBOND c14 c13)
    (AROMATICBOND c15 c14)
    (AROMATICBOND c16 c15)
    (AROMATICBOND n4 c16)
    (bond h10 c12)
    (bond h11 c13)
    (bond h12 c14)
    (bond h13 c15)
    (bond h14 c16)
    (bond c12 h10)
    (bond c13 h11)
    (bond c14 h12)
    (bond c15 h13)
    (bond c16 h14)
    (bond o4 cr1)
    (doublebond cr1 o5)
    (doublebond cr1 o6)
    (bond cr1 cl1)
    (bond cr1 o4)
    (doublebond o5 cr1)
    (doublebond o6 cr1)
    (bond cl1 cr1)
    ; second PCC
    (bond n5 h15)
    (bond h15 n5)
    (AROMATICBOND c17 n5)
    (AROMATICBOND c17 c18)
    (AROMATICBOND c18 c19)
    (AROMATICBOND c19 c20)
    (AROMATICBOND c20 c21)
    (AROMATICBOND c21 n5)
    (AROMATICBOND n5 c17)
    (AROMATICBOND c18 c17)
    (AROMATICBOND c19 c18)
    (AROMATICBOND c20 c19)
    (AROMATICBOND c21 c20)
    (AROMATICBOND n5 c21)
    (bond h16 c17)
    (bond h17 c18)
    (bond h18 c19)
    (bond h19 c20)
    (bond h20 c21)
    (bond c17 h16)
    (bond c18 h17)
    (bond c19 h18)
    (bond c20 h19)
    (bond c21 h20)
    (bond o7 cr2)
    (doublebond cr2 o8)
    (doublebond cr2 o9)
    (bond cr2 cl2)
    (bond cr2 o7)
    (doublebond o8 cr2)
    (doublebond o9 cr2)
    (bond cl2 cr2)
    ; third PCC
    (bond n6 h21)
    (bond h21 n6)
    (AROMATICBOND c22 n6)
    (AROMATICBOND c22 c23)
    (AROMATICBOND c23 c24)
    (AROMATICBOND c24 c25)
    (AROMATICBOND c25 c26)
    (AROMATICBOND c26 n6)
    (AROMATICBOND n6 c22)
    (AROMATICBOND c23 c22)
    (AROMATICBOND c24 c23)
    (AROMATICBOND c25 c24)
    (AROMATICBOND c26 c25)
    (AROMATICBOND n6 c26)
    (bond h22 c22)
    (bond h23 c23)
    (bond h24 c24)
    (bond h25 c25)
    (bond h26 c26)
    (bond c22 h22)
    (bond c23 h23)
    (bond c24 h24)
    (bond c25 h25)
    (bond c26 h26)
    (bond o10 cr3)
    (doublebond cr3 o11)
    (doublebond cr3 o12)
    (bond cr3 cl3)
    (bond cr3 o10)
    (doublebond o11 cr3)
    (doublebond o12 cr3)
    (bond cl3 cr3)
    ; sodium hydroxide Na-OH
    (bond na o50)
    (bond o50 na)
    (bond h50 o50)
    (bond o50 h50)
    ; water H-OH
    (bond h52 o51)
    (bond o51 h52)
    (bond o51 h51)
    (bond h51 o51)
)
(:goal
(and
    (doublebond c1 o1)
    (bond c1 c2)
    (bond c2 c3)
    (bond c3 c4)
    (bond c4 c5)
    (bond c5 c6)
    (bond c6 c1)
    (bond c6 c27)
    (bond c27 c11)
    (bond c11 c10)
    (bond c27 c7)
    (bond c7 c8)
    (doublebond c8 o2)
    (bond c8 c9)
    (bond c2 h3)
    (bond c3 h4)
    (bond c3 h5)
    (bond c4 h6)
    (bond c4 h7)
    (bond c5 h8)
    (bond c5 h27)
    (bond c6 h29)
    (bond c27 h45)
    (bond c11 h42)
    (bond c11 h43)
    (bond c10 h39)
    (bond c10 h40)
    (bond c10 h41)
    (bond c7 h28)
    (bond c7 h33)
    (bond c9 h34)
    (bond c9 h35)
    (bond c9 h36)
)
)
)