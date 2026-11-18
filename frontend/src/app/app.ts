import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  selector: 'app-root',
  styleUrl: './app.css',
  templateUrl: './app.html',
})
export class App {
  protected readonly pages = [
    { path: '/', label: 'Chat', exact: true },
    { path: '/evaluation', label: 'Evaluation', exact: false },
    { path: '/performance', label: 'Performance', exact: false },
    { path: '/usage', label: 'Usage', exact: false },
    { path: '/mcp', label: 'MCP', exact: false },
  ];
}
