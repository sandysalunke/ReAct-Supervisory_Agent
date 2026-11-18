import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', loadComponent: () => import('./chat/chat').then((m) => m.Chat) },
  {
    path: 'evaluation',
    loadComponent: () => import('./evaluation/evaluation').then((m) => m.Evaluation),
  },
  {
    path: 'performance',
    loadComponent: () => import('./performance/performance').then((m) => m.Performance),
  },
  { path: 'usage', loadComponent: () => import('./usage/usage').then((m) => m.Usage) },
  { path: 'mcp', loadComponent: () => import('./mcp/mcp').then((m) => m.Mcp) },
  { path: '**', redirectTo: '' },
];
