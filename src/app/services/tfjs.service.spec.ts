import { TestBed } from '@angular/core/testing';

import { TfjsService } from './tfjs.service';

describe('TfjsService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: TfjsService = TestBed.get(TfjsService);
    expect(service).toBeTruthy();
  });
});
