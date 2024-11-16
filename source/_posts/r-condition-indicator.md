---
title: Source Code for Extracting CIs
date: 2020-03-18 12:14:38
categories:
- Etc.
tags:
- Statistics
- R
---
# CIs?

> Condition Indicator의 약자로써, Feature extraction의 과정이며 Smart data와 유사하다.

<!-- More -->

***

# Window size에 따라 CI를 구하는 함수

~~~R
sk=function(x,Winsiz){
	i=0
	sk=0
	while(i<floor(length(x)/Winsiz)){
		j=1
		sam=0
		while(j<=Winsiz){
			sam[j]=x[i*Winsiz+j]
			j=j+1
		}
		sk[i+1]<-sum(((sam-mean(sam))^3)/sd(sam)^3)*(1/(Winsiz-1))
		i=i+1
	}
	return(sk)
}

ku=function(x,Winsiz){
	i=0
	ku=0
	while(i<floor(length(x)/Winsiz)){
		j=1
		sam=0
		while(j<=Winsiz){
			sam[j]=x[i*Winsiz+j]
			j=j+1
		}
		ku[i+1]<-sum(((sam-mean(sam))^4)/sd(sam)^4)*(1/(Winsiz-1))
		i=i+1
	}
	return(ku)
}

rm=function(x,Winsiz){
	i=0
	rm=0
	while(i<floor(length(x)/Winsiz)){
		j=1
		sam=0
		while(j<=Winsiz){
			sam[j]=x[i*Winsiz+j]
			j=j+1
		}
		rm[i+1]<-sqrt(sum(sam^2)/Winsiz)
		i=i+1
	}
	return(rm)
}

p2p=function(x,Winsiz){
  i=0
  pp=0
  while(i<floor(length(x)/Winsiz)){
    j=1
    sam=0
    while(j<=Winsiz){
      sam[j]=x[i*Winsiz+j]
      j=j+1
    }
    pp[i+1]<-max(sam)-min(sam)
    i=i+1
  }
  return(pp)
}

iq=function(x,Winsiz){
  i=0
  qq=0
  while(i<floor(length(x)/Winsiz)){
    j=1
    sam=0
    while(j<=Winsiz){
      sam[j]=x[i*Winsiz+j]
      j=j+1
    }
    qq[i+1]<-IQR(sam)
    i=i+1
  }
  return(qq)
}

cf=function(x,Winsiz){
  i=0
  cc=0
  while(i<floor(length(x)/Winsiz)){
    j=1
    sam=0
    while(j<=Winsiz){
      sam[j]=x[i*Winsiz+j]
      j=j+1
    }
    cc[i+1]<-(max(sam)-min(sam))/(sqrt(sum(sam^2)/Winsiz))
    i=i+1
  }
  return(cc)
}
~~~

| 함수(변수)명 |            뜻            |
| :----------: | :----------------------: |
|     sk()     |      Skewness(왜도)      |
|     ku()     |      Kurtosis(첨도)      |
|     rm()     |           RMS            |
|    p2p()     |       Peak to peak       |
|     iq()     |           IQR            |
|     cf()     |       Crest factor       |
|    Winsiz    |       Window size        |
|     sam      | Window size만큼의 Sample |

+ Moving average 이용
+ Real-time에 적용하기 용이

~~~R
sk=function(x,Winsiz){
  i=0
  sk=0
  while((i+Winsiz)<=length(x)){
    j=1
    sam=0
    while(j<=Winsiz){
      sam[j]=x[i+j]
      j=j+1
    }
    sk[i+1]<-sum(((sam-mean(sam))^3)/sd(sam)^3)*(1/(Winsiz-1))
    i=i+1
  }
  return(sk)
}

ku=function(x,Winsiz){
  i=0
  ku=0
  while((i+Winsiz)<=length(x)){
    j=1
    sam=0
    while(j<=Winsiz){
      sam[j]=x[i+j]
      j=j+1
    }
    ku[i+1]<-sum(((sam-mean(sam))^4)/sd(sam)^4)*(1/(Winsiz-1))
    i=i+1
  }
  return(ku)
}

rm=function(x,Winsiz){
  i=0
  rm=0
  while((i+Winsiz)<=length(x)){
    j=1
    sam=0
    while(j<=Winsiz){
      sam[j]=x[i+j]
      j=j+1
    }
    rm[i+1]<-sqrt(sum(sam^2)/Winsiz)
    i=i+1
  }
  return(rm)
}


p2p=function(x,Winsiz){
  i=0
  pp=0
  while((i+Winsiz)<=length(x)){
    j=1
    sam=0
    while(j<=Winsiz){
      sam[j]=x[i+j]
      j=j+1
    }
    pp[i+1]<-max(sam)-min(sam)
    i=i+1
  }
  return(pp)
}

iq=function(x,Winsiz){
  i=0
  qq=0
  while((i+Winsiz)<=length(x)){
    j=1
    sam=0
    while(j<=Winsiz){
      sam[j]=x[i+j]
      j=j+1
    }
    qq[i+1]<-IQR(sam)
    i=i+1
  }
  return(qq)
}

cf=function(x,Winsiz){
  i=0
  cc=0
  while((i+Winsiz)<=length(x)){
    j=1
    sam=0
    while(j<=Winsiz){
      sam[j]=x[i+j]
      j=j+1
    }
    cc[i+1]<-(max(sam)-min(sam))/(sqrt(sum(sam^2)/Winsiz))
    i=i+1
  }
  return(cc)
}
~~~

+ 위 함수도 동일한 맥락
+ 하지만 Moving average를 어떤식으로 이용했는지에 차이
+ Window size의 움직임을 Window size(위) 혹은 주기(아래)에 기준을 둔 차이

***

# MakeData

~~~R
MakeData=function(data,Winsiz,name){
  X_skew=sk(data$X,Winsiz)
  X_kurt=ku(data$X,Winsiz)
  X_rms=rm(data$X,Winsiz)
  X_p2p=p2p(data$X,Winsiz)
  X_iq=iq(data$X,Winsiz)
  X_cf=cf(data$X,Winsiz)
  Y_skew=sk(data$Y,Winsiz)
  Y_kurt=ku(data$Y,Winsiz)
  Y_rms=rm(data$Y,Winsiz)
  Y_p2p=p2p(data$Y,Winsiz)
  Y_iq=iq(data$Y,Winsiz)
  Y_cf=cf(data$Y,Winsiz)
  Z_skew=sk(data$Z,Winsiz)
  Z_kurt=ku(data$Z,Winsiz)
  Z_rms=rm(data$Z,Winsiz)
  Z_p2p=p2p(data$Z,Winsiz)
  Z_iq=iq(data$Z,Winsiz)
  Z_cf=cf(data$Z,Winsiz)
  all=cbind(X_skew,X_kurt,X_rms,X_p2p,X_iq,X_cf,Y_skew,Y_kurt,Y_rms,Y_p2p,Y_iq,Y_cf,Z_skew,Z_kurt,Z_rms,Z_p2p,Z_iq,Z_cf)
  options(max.print=10000000)
  print("Im writing now")
  write.table(all[,1:18],name,sep='\t',row.names=F)
  return(all)
}
~~~

| 함수(변수)명 |                            뜻                             |
| :----------: | :-------------------------------------------------------: |
|  MakeData()  | CI를 각 축에 대해 Window size에 맞춰 구하고 저장하는 함수 |
|     data     |               DAQ를 이용해 구한 Data를 입력               |
|     all      |          cbind()함수를 통해 직접 구한 CIs를 결합          |

+ `.txt`파일로 Export
+ CIs table을 return

~~~R
asdf=5000

f1=MakeData(fault6,asdf,'Fault6.txt')
f2=MakeData(fault7,asdf,'Fault7.txt')
f3=MakeData(fault8,asdf,'Fault8.txt')
f4=MakeData(fault9,asdf,'Fault9.txt')

f5=rbind(f1,f2,f3,f4)
write.table(f5[,1:18],'fault.txt',sep='\t',row.names=F)

h1=MakeData(normal5,asdf,'Normal5.txt')
h2=MakeData(normal6,asdf,'Normal6.txt')
h3=MakeData(normal7,asdf,'Normal7.txt')
h4=MakeData(normal8,asdf,'Normal8.txt')

h5=rbind(h1,h2,h3,h4)
write.table(h5[,1:18],'health.txt',sep='\t',row.names=F)
~~~

+ 이런식으로 사용

***

# FDR

![fdr](/images/r-condition-indicator/fdr.png)

~~~R
fdr=function(data1,data2){
  ff=(mean(data1)-mean(data2))^2/((sd(data1))^2+(sd(data2))^2)
  return(ff)
}

allfdr=function(data1,data2){
  i=1
  while(i<=ncol(data1)){
    cat(i,' : ',fdr(data1[,i],data2[,i]),'\n')
    i=i+1
  }
}
~~~

| 함수(변수)명 |            뜻            |
| :----------: | :----------------------: |
|    fdr()     | FDR식을 이용한 수치 계산 |
|   allfdr()   |   모든 CIs의 FDR 출력    |
| data1, data2 |  FDR에 이용할 Data set   |

***

# FDR을 이용해 Window size 결정

~~~R
test1=function(data,Winsiz){
  Z_kurt=ku(data$Z,Winsiz)
  Z_rms=rm(data$Z,Winsiz)
  Z_iq=iq(data$Z,Winsiz)
  Z_cf=cf(data$Z,Winsiz)
  all=cbind(Z_kurt,Z_rms,Z_iq,Z_cf)
  options(max.print=10000000)
  return(all)
}

FDRtest=function(max1,size,start){
  asdf=start
  i=1
  ku1=0
  rm1=0
  iq1=0
  cf1=0
  ti1=0
  while(asdf<=max1){
    f1=test1(fault6,asdf)
    f2=test1(fault7,asdf)
    f3=test1(fault8,asdf)
    f4=test1(fault9,asdf)
    
    f5=rbind(f1,f2,f3,f4)
    
    h1=test1(normal5,asdf)
    h2=test1(normal6,asdf)
    h3=test1(normal7,asdf)
    h4=test1(normal8,asdf)
    
    h5=rbind(h1,h2,h3,h4)
    
    ku1[i]<-fdr(f5[,1],h5[,1])
    rm1[i]<-fdr(f5[,2],h5[,2])
    iq1[i]<-fdr(f5[,3],h5[,3])
    cf1[i]<-fdr(f5[,4],h5[,4])
    ti1[i]<-asdf
    
    print(asdf)
    
    if(i%%2==1){
      cat((asdf-start)/(max1-start)*100,'%\n')
    }
    
    asdf=asdf+size
    i=i+1
  }
  gg=cbind(ku1,rm1,iq1,cf1)
  return(gg)
}
~~~

| 함수(변수)명 |          뜻          |
| :----------: | :------------------: |
|   test1()    | MakeData의 축소 형태 |
|  fdrtest()   | 모든 CIs의 FDR 출력  |
|     max1     |         끝점         |
|     size     |  Window size 증가폭  |
|    start     |        시작점        |
|     asdf     |  동적인 Window size  |
|     ku1      |   FDR of Kurtosis    |
|     rm1      |      FDR of RMS      |
|     iq1      |      FDR of IQR      |
|     cf1      |      FDR of CF       |
|     ti1      |   Window size 배열   |

~~~R
as=FDRtest(100000,100,1000)
~~~

<img src="/images/r-condition-indicator/fdr-graph.png" alt="fdr-graph" width="1024" />

***

# 확률밀도함수

~~~R
plot(density(f5[,17]),xlab="Magnitude", ylab="Density", main="Z_iq", col="red")
par(new=TRUE)
lines(density(h5[,17]),col='blue')
~~~

+ `xlim=c()`, `ylim=c()`로 스케일 조정 가능

![z-iq](/images/r-condition-indicator/z-iq.png)