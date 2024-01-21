import{s as Ce,i as V,j as _e,n as we}from"../chunks/scheduler.9b9e513e.js";import{S as Me,i as Te,r as Le,u as He,v as ye,d as be,t as ke,w as Pe,g as a,s as u,h as s,A as o,c as m,k as I,a as l,f as i}from"../chunks/index.781c9930.js";import{P as je,g as Ae,a as xe}from"../chunks/post_layout.601f0e77.js";function ze(E){let n,v='<a href="#motivation">Motivation</a>',d,f,r='It seems <a href="https://www.instagram.com/reel/Cv5SkdLrVoq/?igsh=MXIzaDgzODBxZDlsZg==" rel="nofollow noopener noreferrer external" target="_blank">I am not the only one</a> that find some English town names slightly quirky or outright funny. Maybe it is because their spelling is quite different to the rest of English vocabulary we use one our daily lives, or maybe some of these places were named a long time ago when English looked and sounded very different from today.',h,p,X="Whatever the reason, I thought it would be fun to train a small transformer to come up with similar-sounding names, if only to see if we would fall for a deepfake sign on a road.",q,c,ee='<a href="#data">Data</a>',$,C,te="Relevant data for most countries can be easily accessed from <code>http://download.geonames.org</code>, and the data contain administrative information too, so it is possible to use data for a single state or even county. In the case of England, available data contains a bit over 16,000 names.",D,_,le='<a href="#architecture">Architecture</a>',S,w,ie='This project was done with Keras NLP. The model is a transformer decoder, identical to the one in <a href="https://arxiv.org/abs/1706.03762" rel="nofollow noopener noreferrer external" target="_blank">Attention Is All You Need</a>, with the following parameters',U,M,ne="<li>Attention heads: 3</li> <li>Transformer layers: 3</li> <li>Fully connected layer size: 512</li> <li>Embedding size: 32</li>",N,T,ae="Training parameters were:",Z,L,se="<li>Batch size: 32</li> <li>Epochs: 200</li> <li>Optimizer: ADAM with <code>learning_rate = 0.001</code></li>",G,H,oe="All of this took around 10 minutes on a CPU.",K,y,re="The tokenizer worked at character level, which for English produced 30 characters, including beginning and end of text tokens",R,x,ue='<a href="#results">Results</a>',B,b,me="Here are some of the generated names I liked the most for England",F,k,he="<li>Upminster</li> <li>Whippleigh</li> <li>Kelingbrough</li> <li>Millers mill</li> <li>Croomfleet</li> <li>Chillarton</li> <li>Egerton on the hill</li> <li>Kilkinster</li> <li>Ashton Dingley</li> <li>Hegleton</li>",O,P,fe="As a non-native speaker I have to admit I wouldn’t bat an eye if I saw any of those on a road sign (Perhap’s Miller’s Mill would raise some suspicion), which somehow makes it more amusing.",W,j,de="This little experiment can be re-run for any other country or subregion with barely any change in command line parameters, so I also had a chuckle doing this for Mexican towns. Some of the best ones:",Y,A,pe="<li>San Juan Guilalapam</li> <li>El Malo</li> <li>Llano Grande</li> <li>Yuchiqui de la Luma</li> <li>Quinicuelo</li>",J,g,ve='<a href="#code">Code</a>',Q,z,ce='The project’s repo is <a href="https://github.com/nestorSag/towngen" rel="nofollow noopener noreferrer external" target="_blank">here</a>.';return{c(){n=a("h2"),n.innerHTML=v,d=u(),f=a("p"),f.innerHTML=r,h=u(),p=a("p"),p.textContent=X,q=u(),c=a("h2"),c.innerHTML=ee,$=u(),C=a("p"),C.innerHTML=te,D=u(),_=a("h2"),_.innerHTML=le,S=u(),w=a("p"),w.innerHTML=ie,U=u(),M=a("ul"),M.innerHTML=ne,N=u(),T=a("p"),T.textContent=ae,Z=u(),L=a("ul"),L.innerHTML=se,G=u(),H=a("p"),H.textContent=oe,K=u(),y=a("p"),y.textContent=re,R=u(),x=a("h2"),x.innerHTML=ue,B=u(),b=a("p"),b.textContent=me,F=u(),k=a("ul"),k.innerHTML=he,O=u(),P=a("p"),P.textContent=fe,W=u(),j=a("p"),j.textContent=de,Y=u(),A=a("ul"),A.innerHTML=pe,J=u(),g=a("h1"),g.innerHTML=ve,Q=u(),z=a("p"),z.innerHTML=ce,this.h()},l(e){n=s(e,"H2",{id:!0,"data-svelte-h":!0}),o(n)!=="svelte-1qh4qj0"&&(n.innerHTML=v),d=m(e),f=s(e,"P",{"data-svelte-h":!0}),o(f)!=="svelte-rjmjst"&&(f.innerHTML=r),h=m(e),p=s(e,"P",{"data-svelte-h":!0}),o(p)!=="svelte-1135iis"&&(p.textContent=X),q=m(e),c=s(e,"H2",{id:!0,"data-svelte-h":!0}),o(c)!=="svelte-14ht7vg"&&(c.innerHTML=ee),$=m(e),C=s(e,"P",{"data-svelte-h":!0}),o(C)!=="svelte-og9z6v"&&(C.innerHTML=te),D=m(e),_=s(e,"H2",{id:!0,"data-svelte-h":!0}),o(_)!=="svelte-m3c4bv"&&(_.innerHTML=le),S=m(e),w=s(e,"P",{"data-svelte-h":!0}),o(w)!=="svelte-1gwcci8"&&(w.innerHTML=ie),U=m(e),M=s(e,"UL",{"data-svelte-h":!0}),o(M)!=="svelte-zoyk4i"&&(M.innerHTML=ne),N=m(e),T=s(e,"P",{"data-svelte-h":!0}),o(T)!=="svelte-1daohs7"&&(T.textContent=ae),Z=m(e),L=s(e,"UL",{"data-svelte-h":!0}),o(L)!=="svelte-9yb2jp"&&(L.innerHTML=se),G=m(e),H=s(e,"P",{"data-svelte-h":!0}),o(H)!=="svelte-18xe92u"&&(H.textContent=oe),K=m(e),y=s(e,"P",{"data-svelte-h":!0}),o(y)!=="svelte-18cgx3r"&&(y.textContent=re),R=m(e),x=s(e,"H2",{id:!0,"data-svelte-h":!0}),o(x)!=="svelte-xvlaxe"&&(x.innerHTML=ue),B=m(e),b=s(e,"P",{"data-svelte-h":!0}),o(b)!=="svelte-1h17g8j"&&(b.textContent=me),F=m(e),k=s(e,"UL",{"data-svelte-h":!0}),o(k)!=="svelte-o1jbnn"&&(k.innerHTML=he),O=m(e),P=s(e,"P",{"data-svelte-h":!0}),o(P)!=="svelte-c5q24y"&&(P.textContent=fe),W=m(e),j=s(e,"P",{"data-svelte-h":!0}),o(j)!=="svelte-1qpls0m"&&(j.textContent=de),Y=m(e),A=s(e,"UL",{"data-svelte-h":!0}),o(A)!=="svelte-5cyzm3"&&(A.innerHTML=pe),J=m(e),g=s(e,"H1",{id:!0,"data-svelte-h":!0}),o(g)!=="svelte-azjqur"&&(g.innerHTML=ve),Q=m(e),z=s(e,"P",{"data-svelte-h":!0}),o(z)!=="svelte-1ask3tc"&&(z.innerHTML=ce),this.h()},h(){I(n,"id","motivation"),I(c,"id","data"),I(_,"id","architecture"),I(x,"id","results"),I(g,"id","code")},m(e,t){l(e,n,t),l(e,d,t),l(e,f,t),l(e,h,t),l(e,p,t),l(e,q,t),l(e,c,t),l(e,$,t),l(e,C,t),l(e,D,t),l(e,_,t),l(e,S,t),l(e,w,t),l(e,U,t),l(e,M,t),l(e,N,t),l(e,T,t),l(e,Z,t),l(e,L,t),l(e,G,t),l(e,H,t),l(e,K,t),l(e,y,t),l(e,R,t),l(e,x,t),l(e,B,t),l(e,b,t),l(e,F,t),l(e,k,t),l(e,O,t),l(e,P,t),l(e,W,t),l(e,j,t),l(e,Y,t),l(e,A,t),l(e,J,t),l(e,g,t),l(e,Q,t),l(e,z,t)},p:we,d(e){e&&(i(n),i(d),i(f),i(h),i(p),i(q),i(c),i($),i(C),i(D),i(_),i(S),i(w),i(U),i(M),i(N),i(T),i(Z),i(L),i(G),i(H),i(K),i(y),i(R),i(x),i(B),i(b),i(F),i(k),i(O),i(P),i(W),i(j),i(Y),i(A),i(J),i(g),i(Q),i(z))}}}function Ee(E){let n,v;const d=[E[0],ge];let f={$$slots:{default:[ze]},$$scope:{ctx:E}};for(let r=0;r<d.length;r+=1)f=V(f,d[r]);return n=new je({props:f}),{c(){Le(n.$$.fragment)},l(r){He(n.$$.fragment,r)},m(r,h){ye(n,r,h),v=!0},p(r,[h]){const p=h&1?Ae(d,[h&1&&xe(r[0]),h&0&&xe(ge)]):{};h&2&&(p.$$scope={dirty:h,ctx:r}),n.$set(p)},i(r){v||(be(n.$$.fragment,r),v=!0)},o(r){ke(n.$$.fragment,r),v=!1},d(r){Pe(n,r)}}}const ge={title:"A Generative Model For English Town Names",tags:["NLP","generative models"],image_caption:"Styles across one of the dimensions in latent space",image:"/town-name-generator/images/sign.jpg",created:"2024-01-18T00:00:00.000Z",updated:"2024-01-21T18:56:38.178Z",images:[],slug:"/town-name-generator/+page.md",path:"/town-name-generator",toc:[{depth:2,title:"Motivation",slug:"motivation"},{depth:2,title:"Data",slug:"data"},{depth:2,title:"Architecture",slug:"architecture"},{depth:2,title:"Results",slug:"results"},{depth:1,title:"Code",slug:"code"}]};function Ie(E,n,v){return E.$$set=d=>{v(0,n=V(V({},n),_e(d)))},n=_e(n),[n]}class Se extends Me{constructor(n){super(),Te(this,n,Ie,Ee,Ce,{})}}export{Se as component};
