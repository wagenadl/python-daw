// pyphyc.cpp

#include <QPainter>
#include <QPolygon>
#include <stdint.h>
#include <iostream>

#ifdef _WIN32
#define DECLSPEC __declspec(dllexport)
#else
#define DECLSPEC

extern "C" {
  int foo(int x) {
    return x*x;
  }
  

DECLSPEC  void quickdraw(QPainter *ptr,
                 int16_t *data,
                 int N,
                 int stride,
                 int x0,
                 int w,
                 int y0,
                 float vscale) {
    if (N>2*w) {
      QPolygon pp(2*w);
      for (int x=0; x<w; x++) {
        int n0 = N*x/w;
        int n1 = N*(x+1)/w;
        if (n1==n0)
          n1++;
        int v0 = data[n0*stride];
        int v1 = data[n0*stride];
        for (int n=n0+1; n<n1; n++) {
          int v = data[n*stride];
          if (v<v0)
            v0 = v;
          else if (v>v1)
            v1 = v;
        }
        pp[x] = QPoint(x0 + x, y0 + v0*vscale);
        pp[2*w-1-x] = QPoint(x0 + x, y0 + v1*vscale);
      }
      ptr->drawPolygon(pp);
    } else {
      QPolygon pp(N);
      for (int n=0; n<N; n++) 
        pp[n] = QPoint(x0 + w*n/N, y0 + vscale*data[n*stride]);
      ptr->drawPolyline(pp);
    }
  }
}

               
