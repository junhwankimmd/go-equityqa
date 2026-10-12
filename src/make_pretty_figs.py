#!/usr/bin/env python
"""Polished, precisely-aligned SVG schematics -> 300 dpi PNG + vector PDF (cairosvg).
Figure 1 (workflow) and Figure 4 (failure -> mitigation)."""
import cairosvg
FONT="Arial, Helvetica, sans-serif"
INK="#1F2A37"; SUB="#5B6773"; BORD="#D8DCE2"
C={"blue":"#2E6FA7","green":"#2E8B6B","purple":"#7A5AA0","amber":"#C58A2E",
   "red":"#C1553B","slate":"#566575","orange":"#E0932A"}
TINT={"blue":"#ECF3F9","green":"#EAF4EF","purple":"#F2ECF7","amber":"#F9F1E3",
      "red":"#F8ECE7","slate":"#EEF1F4","orange":"#FCF2E1"}
_ID=[0]

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def rrect(x,y,w,h,r,fill,stroke=None,sw=1):
    s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" fill="{fill}"'
    if stroke: s+=f' stroke="{stroke}" stroke-width="{sw}"'
    return s+"/>"
def txt(x,y,s,size,fill=INK,w="normal",anc="start",ls="0"):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'font-weight="{w}" text-anchor="{anc}" letter-spacing="{ls}">{esc(s)}</text>')
def mtxt(x,y,lines,size,fill,lh,anc="start",w="normal"):
    return "".join(txt(x,y+i*lh,l,size,fill,w,anc) for i,l in enumerate(lines))
def badge(cx,cy,n,color):
    return f'<circle cx="{cx}" cy="{cy}" r="10.5" fill="{color}"/>'+txt(cx,cy+3.8,str(n),11.5,"#FFFFFF","bold","middle")
def chevron(cx,cy,color,w=7,h=9,sw=2.4):
    return (f'<polyline points="{cx-w/2},{cy-h/2} {cx+w/2},{cy} {cx-w/2},{cy+h/2}" fill="none" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')
def arrow(x1,y1,x2,y2,color=C["slate"],wd=1.6,head=5.5):
    import math
    ang=math.atan2(y2-y1,x2-x1); bx,by=x2-head*1.5*math.cos(ang),y2-head*1.5*math.sin(ang)
    return (f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="{color}" stroke-width="{wd}" stroke-linecap="round"/>'
            f'<polygon points="{x2},{y2} {bx-head*math.sin(ang):.1f},{by+head*math.cos(ang):.1f} '
            f'{bx+head*math.sin(ang):.1f},{by-head*math.cos(ang):.1f}" fill="{color}"/>')

def vcenter(y,h,tsize,n,sublh,sub_size=10.0,gap_factor=1.3):
    """return (title_baseline_y, first_subtitle_baseline_y) so the title+n subtitle
    lines sit vertically centred in the card [y, y+h]."""
    TA=tsize*0.72; GAP=tsize*gap_factor; sd=sub_size*0.28
    block=TA+((GAP+(n-1)*sublh+sd) if n else tsize*0.28)
    top=(h-block)/2; ty=y+top+TA
    return ty, ty+GAP
def card(defs,x,y,w,h,accent,r=12,rail=6,fill="#FFFFFF"):
    """rounded card with a left accent rail clipped to the rounded corners."""
    _ID[0]+=1; cid=f"clip{_ID[0]}"
    defs.append(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}"/></clipPath>')
    return (rrect(x,y,w,h,r,fill,BORD,1.2)
            + f'<g clip-path="url(#{cid})"><rect x="{x}" y="{y}" width="{rail}" height="{h}" fill="{accent}"/></g>')

def build(w,h,defs,body,base,w_px):
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}">'
         f'<rect width="{w}" height="{h}" fill="#FFFFFF"/><defs>{"".join(defs)}</defs>{"".join(body)}</svg>')
    open(f"results/{base}.svg","w").write(svg)
    cairosvg.svg2png(bytestring=svg.encode(),write_to=f"results/{base}.png",output_width=w_px)
    cairosvg.svg2pdf(bytestring=svg.encode(),write_to=f"results/{base}.pdf")
    print("saved",base)

# ================= FIGURE 1: WORKFLOW (vertical) =================
W,H=520,512; defs=[]; b=[]
# preregistration ribbon
b.append(rrect(24,16,W-48,28,8,TINT["amber"],C["amber"],1))
b.append(f'<circle cx="44" cy="30" r="4.3" fill="{C["amber"]}"/>')
b.append(txt(57,34,"Preregistered before confirmatory data collection",11,C["amber"],"bold"))
cards=[("blue","308-item benchmark",["5 arms: provenance, invariance,","erasure, overconfidence, reasoning"]),
       ("green","3 frontier models",["GPT-5.5  ·  Claude Opus 4.8","Gemini 3.1 Pro Preview"]),
       ("purple","Automated scoring",["3-layer citation check (exists →","identity → support) + judges"]),
       ("slate","Analysis",["mixed-effects models, human κ,","judge-model sensitivity"])]
cx0,cw,ch=40,262,72; ys=[66,172,278,384]; colx=cx0+cw/2
for i,((col,title,lines),y) in enumerate(zip(cards,ys)):
    b.append(card(defs,cx0,y,cw,ch,C[col]))
    ty,fy=vcenter(y,ch,12,len(lines),13.5,sub_size=10)
    b.append(badge(cx0+26,ty-4.3,i+1,C[col]))
    b.append(txt(cx0+44,ty,title,12,INK,"bold"))
    b.append(mtxt(cx0+20,fy,lines,9.8,SUB,13.5))
    if i<3: b.append(arrow(colx,y+ch,colx,ys[i+1],C["slate"],1.8,6))
# coverage proxy: side input into scoring + analysis
covx,covw,covh=328,164,86; covy=(ys[2]+ys[3]+ch)/2-covh/2
b.append(card(defs,covx,covy,covw,covh,C["orange"]))
cty,cfy=vcenter(covy,covh,11,2,13,sub_size=9.3)
b.append(txt(covx+covw/2+3,cty,"Coverage proxy",11,C["orange"],"bold","middle"))
b.append(mtxt(covx+covw/2+3,cfy,["PubMed articles per","topic × region, log10(count+1)"],9.3,SUB,13,"middle"))
b.append(arrow(covx,covy+covh*0.30,cx0+cw,ys[2]+ch/2,C["orange"],1.4,5))
b.append(arrow(covx,covy+covh*0.70,cx0+cw,ys[3]+ch/2,C["orange"],1.4,5))
# footer
b.append(f'<line x1="24" y1="470" x2="{W-24}" y2="470" stroke="{BORD}" stroke-width="1"/>')
b.append(mtxt(W/2,486,["Outcomes:  knowledge (accuracy)  ·  provenance (citation support)","·  behavior (nationality, erasure)"],10,SUB,13,"middle"))
build(W,H,defs,b,"fig1_workflow",1560)

# ================= FIGURE 4: FAILURE -> MITIGATION =================
W,H=760,418; defs=[]; b=[]
PX,PW=24,300; SX,SW,SH=408,328,54
b.append(txt(PX+PW/2,26,"FAILURES IDENTIFIED",11,C["slate"],"bold","middle","0.6"))
b.append(txt(SX+SW/2,26,"MITIGATIONS",11,C["slate"],"bold","middle","0.6"))
def lane(py,ph,pcol,ptitle,plines,scol,sols):
    b.append(card(defs,PX,py,PW,ph,C[pcol],fill=TINT[pcol]))
    pty,pfy=vcenter(py,ph,13,len(plines),15,sub_size=10.3)
    b.append(txt(PX+22,pty,ptitle,13,C[pcol],"bold"))
    b.append(mtxt(PX+22,pfy,plines,10.3,INK,15))
    sys=[py, py+ph-SH]
    for (t,ls),sy in zip(sols,sys):
        b.append(card(defs,SX,sy,SW,SH,C[scol]))
        sty,sfy=vcenter(sy,SH,11,len(ls),12.5,sub_size=9.6)
        b.append(txt(SX+20,sty,t,11,C[scol],"bold"))
        b.append(mtxt(SX+20,sfy,ls,9.6,SUB,12.5))
        b.append(arrow(PX+PW+7,py+ph/2,SX-7,sy+SH/2,C["slate"],1.5,5))
lane(46,120,"red","Provenance failure",
     ["At low coverage, models cite a real,","on-topic paper that does not contain","the specific claimed fact (not an","outright fabrication)."],
     "blue",[("Retrieval + support verification",["check existence, record identity,","and support"]),
             ("Full-text-grounded citing",["abstain when coverage is low or","support is unverifiable"])])
lane(208,120,"purple","Behavioral bias",
     ["Models infer resource scarcity from","a patient's nationality and change an","otherwise-invariant recommendation."],
     "green",[("State the resource setting",["give the care context explicitly","in the prompt (low-cost control)"]),
              ("Counterfactual invariance audit",["swap nationality and flag any","divergence before deployment"])])
# cross-cutting band
b.append(card(defs,24,350,W-48,52,C["slate"],fill=TINT["slate"]))
cty,cfy=vcenter(350,52,9.5,2,13,sub_size=9.8)
b.append(txt(44,cty,"ACROSS BOTH",9.5,C["slate"],"bold","start","0.6"))
b.append(mtxt(44,cfy,["Score hallucination by whether a citation supports its claim, not merely exists   ·   stratify evaluation by",
                      "literature coverage and geography   ·   build participatory, equity-focused benchmarks"],9.8,INK,13))
build(W,H,defs,b,"fig4_mitigations",2280)
print("done")
