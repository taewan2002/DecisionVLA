// Native Figma Plugin API source for the README architecture figure.
// Run through use_figma in the linked Design file. The frame/photo are prepared separately.

const page = await figma.getNodeByIdAsync('1836:15');
await figma.setCurrentPageAsync(page);
await Promise.all(['Regular','Medium','Semi Bold','Bold'].map(style => figma.loadFontAsync({family:'Inter',style})));
const root = await figma.getNodeByIdAsync('13366:92873');
const image = await figma.getNodeByIdAsync('13366:92874');
if (root.children.some(n => n.name === 'DecisionVLA / title')) return {alreadyBuilt:true,rootNodeId:root.id};
const ids=[], colors={ink:'#182433',muted:'#536376',line:'#52657A',border:'#C3CEDB',blue:'#E7F2FA',blueInk:'#28648D',purple:'#F2EDFF',purpleInk:'#6F4EAC',green:'#EAF6ED',greenInk:'#2C6B44',orange:'#FFF1DB',orangeInk:'#A76B19',white:'#FFFFFF'};
function rgb(hex){const h=hex.replace('#','');return {r:parseInt(h.slice(0,2),16)/255,g:parseInt(h.slice(2,4),16)/255,b:parseInt(h.slice(4,6),16)/255};}
function paint(hex){return [{type:'SOLID',color:rgb(hex)}];}
function keep(n){ids.push(n.id);return n;}
function text(parent,name,value,size=20,weight='Regular',color=colors.ink,width){
  const t=keep(figma.createText()); parent.appendChild(t); t.name=name;
  t.fontName={family:'Inter',style:weight}; t.fontSize=size;
  t.lineHeight={unit:'PERCENT',value:130}; t.fills=paint(color); t.characters=value;
  if(width){t.resize(width,30);t.textAutoResize='HEIGHT';}else t.textAutoResize='WIDTH_AND_HEIGHT';
  return t;
}
function label(name,value,x,y,width,size=20,weight='Regular',color=colors.ink){
  const t=text(root,name,value,size,weight,color,width);t.x=x;t.y=y;return t;
}
function rect(parent,name,x,y,w,h,fill,stroke,radius=12){
  const n=keep(figma.createRectangle());parent.appendChild(n);n.set({name,x,y,width:w,height:h,cornerRadius:radius,fills:paint(fill),strokes:stroke?paint(stroke):[],strokeWeight:1.5});return n;
}
function card(name,x,y,w,h,fill,stroke,padding=20,gap=8){
  const f=keep(figma.createAutoLayout('VERTICAL',{name,itemSpacing:gap,paddingLeft:padding,paddingRight:padding,paddingTop:padding,paddingBottom:padding,fills:paint(fill),strokes:paint(stroke),strokeWeight:1.5,cornerRadius:14}));
  root.appendChild(f);f.resize(w,h);f.layoutSizingHorizontal='FIXED';f.layoutSizingVertical='FIXED';f.x=x;f.y=y;
  return f;
}
function ctext(parent,name,value,size=20,weight='Regular',color=colors.ink){
  return text(parent,name,value,size,weight,color,parent.width-parent.paddingLeft-parent.paddingRight);
}
function path(name,pts,color=colors.line,weight=2,dashed=false,arrow=false){
  const minX=Math.min(...pts.map(p=>p[0])),minY=Math.min(...pts.map(p=>p[1]));
  const v=keep(figma.createVector());root.appendChild(v);v.name=name;
  v.vectorPaths=[{windingRule:'NONE',data:pts.map((p,i)=>(i?'L ':'M ')+(p[0]-minX)+' '+(p[1]-minY)).join(' ')}];
  v.x=minX;v.y=minY;v.fills=[];v.strokes=paint(color);v.strokeWeight=weight;v.strokeJoin='ROUND';
  if(dashed)v.dashPattern=[7,6];
  if(arrow){
    const e=pts[pts.length-1],p=pts[pts.length-2],ang=Math.atan2(e[1]-p[1],e[0]-p[0]),l=9,a=.48;
    path(name+' / arrowhead',[[e[0]-l*Math.cos(ang-a),e[1]-l*Math.sin(ang-a)],e,[e[0]-l*Math.cos(ang+a),e[1]-l*Math.sin(ang+a)]],color,weight,false,false);
  }return v;
}
function badge(name,value,x,y,w,fill,ink){
 const f=card(name,x,y,w,30,fill,fill,7,0);f.paddingTop=3;f.paddingBottom=3;
 const t=ctext(f,name+' / label',value,16,'Semi Bold',ink);t.textAlignHorizontal='CENTER';return f;
}
// Title and visual key.
label('DecisionVLA / title','DecisionVLA',48,32,900,42,'Bold');
label('DecisionVLA / subtitle','Direct robot actions from a shared diffusion language model',48,92,1160,23,'Regular',colors.muted);
badge('Legend / frozen','Frozen',1400,53,106,colors.blue,colors.blueInk);
badge('Legend / trainable','Trainable',1520,53,130,colors.purple,colors.purpleInk);
path('Header rule',[[48,142],[1712,142]],colors.border,1);
// The backbone contains both execution modes of the same adapted text weights.
rect(root,'DiffusionGemma / boundary',326,174,1006,508,'#FAFBFD',colors.border,20);
label('Backbone / title','DiffusionGemma · 26B-A4B',350,194,840,26,'Semi Bold');
label('Backbone / sharing','Prefix encoder and canvas decoder share text weights + LoRA',350,234,900,19,'Regular',colors.muted);
label('Inputs / title','Observation',48,204,244,23,'Semi Bold');
label('Inputs / image label','RGB image',48,246,244,19,'Medium',colors.muted);
image.set({x:97,y:280,width:148,height:148,cornerRadius:8});
label('Inputs / resolution','agentview · 128 × 128',48,438,244,17,'Regular',colors.muted).textAlignHorizontal='CENTER';
label('Inputs / instruction label','Task instruction',48,482,244,19,'Medium',colors.muted);
const instruction=card('Inputs / instruction',48,516,244,94,'#FFFFFF',colors.border,16,0);
ctext(instruction,'Inputs / instruction example','“Pick up the black bowl\nand place it on the plate.”',18,'Medium');
// Observation encoder.
const vision=card('Encoder / frozen vision',350,284,300,100,colors.blue,'#91B6D1',17,7);
ctext(vision,'Encoder / vision title','Vision encoder',23,'Semi Bold',colors.blueInk);
ctext(vision,'Encoder / vision detail','Native visual projection',18,'Regular',colors.blueInk);
const prefix=card('Encoder / prefix Transformer',350,462,300,108,colors.purple,'#B5A2D6',17,7);
ctext(prefix,'Encoder / prefix title','Prefix Transformer',23,'Semi Bold');
ctext(prefix,'Encoder / prefix detail','Causal text · vision context',18,'Regular',colors.muted);
badge('Encoder / LoRA badge','LoRA',548,432,86,colors.purple,colors.purpleInk);
label('Encoder / once','Encode observation once',350,595,300,19,'Semi Bold',colors.blueInk).textAlignHorizontal='CENTER';
// Immutable observation context reused during the decoding iterations.
const kv=card('Context / KV cache',704,470,164,92,'#EDF2F7','#AABACB',16,5);
ctext(kv,'Context / title','KV cache',23,'Semi Bold').textAlignHorizontal='CENTER';
ctext(kv,'Context / detail','Reuse per step',16,'Regular',colors.muted).textAlignHorizontal='CENTER';
// Action denoising.
const noise=card('Decoder / noisy canvas',922,284,360,90,colors.orange,'#DDC198',16,5);
ctext(noise,'Decoder / canvas title','Noisy action canvas',23,'Semi Bold',colors.orangeInk);
ctext(noise,'Decoder / canvas detail','Uniform init / decreasing re-noise',17,'Regular',colors.orangeInk);
const decoder=card('Decoder / bidirectional Transformer',922,428,360,100,colors.purple,'#B5A2D6',17,7);
ctext(decoder,'Decoder / title','Bidirectional Transformer',23,'Semi Bold');
ctext(decoder,'Decoder / detail','All 56 action slots in parallel',18,'Regular',colors.muted);
badge('Decoder / LoRA badge','LoRA',1180,398,86,colors.purple,colors.purpleInk);
const readout=card('Decoder / native readout',922,572,360,62,colors.purple,'#B5A2D6',16,0);
ctext(readout,'Decoder / readout title','Tied action-token readout',22,'Semi Bold');
label('Decoder / repeat note','4 denoising steps',934,647,348,18,'Semi Bold',colors.orangeInk);
// Control chunk: seven rows, eight columns. Last row denotes the binary gripper.
const output=card('Output / action chunk',1380,360,332,274,colors.green,'#9BC6A7',20,10);
ctext(output,'Output / title','8 × 7 action chunk',25,'Semi Bold',colors.greenInk);
ctext(output,'Output / dimensions','8 timesteps · 7 controls',18,'Regular',colors.greenInk);
const grid=keep(figma.createAutoLayout('VERTICAL',{name:'Output / action matrix',itemSpacing:5,fills:[]}));
output.appendChild(grid);
for(let r=0;r<7;r++){
 const row=keep(figma.createAutoLayout('HORIZONTAL',{name:r===6?'Binary gripper':'Continuous control '+(r+1),itemSpacing:7,fills:[]}));
 grid.appendChild(row);
 for(let c=0;c<8;c++)rect(row,'Action '+(c+1)+' / dimension '+(r+1),0,0,29,12,r===6?'#EBC16E':'#86B993',null,2);
}
ctext(output,'Output / codec','256 bins · binary gripper',18,'Medium',colors.greenInk);
label('Output / execution','Execute chunk → observe again',1360,661,352,18,'Medium',colors.muted).textAlignHorizontal='CENTER';
// Edges are routed around labels and card content.
path('RGB → native vision',[[255,334],[350,334]],colors.line,2,false,true);
path('Vision → prefix',[[500,384],[500,462]],colors.line,2,false,true);
label('Encoder / visual tokens','Visual tokens',511,400,130,16,'Regular',colors.muted);
path('Instruction → prefix',[[292,563],[320,563],[320,533],[350,533]],colors.line,2,false,true);
path('Prefix → KV',[[650,516],[704,516]],colors.line,2,false,true);
path('KV → denoiser',[[868,516],[894,516],[894,478],[922,478]],colors.line,2,false,true);
path('Canvas → denoiser',[[1090,374],[1090,428]],colors.orangeInk,2,false,true);
path('Denoiser → native readout',[[1090,528],[1090,572]],colors.line,2,false,true);
path('Intermediate sample → re-noise',[[1282,603],[1307,603],[1307,329],[1282,329]],colors.orangeInk,2,true,true);
path('Final readout → decoded actions',[[1282,603],[1380,603]],colors.greenInk,2,false,true);
label('Output / final step','Final',1338,575,50,16,'Medium',colors.greenInk);
// Compact explanatory band.
const training=card('Training / denoising objective',48,746,806,146,'#FAFBFD',colors.border,22,12);
ctext(training,'Training / heading','TRAINING · Denoising imitation',21,'Semi Bold');
ctext(training,'Training / corruption','Demo actions → uniform corruption → same denoiser',19,'Regular',colors.muted);
ctext(training,'Training / loss','Cross-entropy over every valid action position',19,'Regular',colors.muted);
const adaptation=card('Training / adapter scope',884,746,828,146,'#FAFBFD',colors.border,22,12);
ctext(adaptation,'Training / LoRA heading','ADAPTATION · Rank-32 LoRA + tied action rows',21,'Semi Bold');
ctext(adaptation,'Training / LoRA scope','Attention Q/K/V/O · dense MLP · routed MoE experts',19,'Regular',colors.muted);
ctext(adaptation,'Training / frozen scope','Base weights, vision and routers remain frozen',19,'Regular',colors.muted);
label('Canvas / implementation note','Native 256-token canvas: 56 action positions + 200 fixed padding positions (masked as attention keys)',350,703,1360,18,'Regular',colors.muted);
root.placeholder=false;
return {rootNodeId:root.id,createdNodeIds:ids,mutatedNodeIds:[root.id,image.id],nodeCount:ids.length,bounds:{width:root.width,height:root.height},textBounds:root.query('TEXT').values(['id','name','x','y','width','height'])};

