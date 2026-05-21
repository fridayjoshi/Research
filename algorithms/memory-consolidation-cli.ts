#!/usr/bin/env node
/**
 * CLI tool for memory consolidation
 *
 * Usage:
 *   memory-consolidation-cli --input memory/2026-05-21.md --memory-file MEMORY.md --budget 10
 *
 * @author Friday
 * @date 2026-05-21
 */

import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from 'fs';
import { join } from 'path';
import {
  MemoryStore,
  consolidate,
  parseDailyLog
} from './memory-consolidation';

interface CLIArgs {
  inputDir?: string;
  inputFile?: string;
  memoryFile: string;
  budgetPerDay: number;
  startDate?: string;
  dryRun: boolean;
  verbose: boolean;
}

function parseArgs(): CLIArgs {
  const args: CLIArgs = {
    memoryFile: 'MEMORY.md',
    budgetPerDay: 10,
    dryRun: false,
    verbose: false
  };

  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];
    const next = process.argv[i + 1];

    switch (arg) {
      case '--input-dir':
        args.inputDir = next;
        i++;
        break;
      case '--input-file':
        args.inputFile = next;
        i++;
        break;
      case '--memory-file':
        args.memoryFile = next;
        i++;
        break;
      case '--budget':
        args.budgetPerDay = parseInt(next, 10);
        i++;
        break;
      case '--start-date':
        args.startDate = next;
        i++;
        break;
      case '--dry-run':
        args.dryRun = true;
        break;
      case '--verbose':
        args.verbose = true;
        break;
      case '--help':
        printHelp();
        process.exit(0);
      default:
        console.error(`Unknown argument: ${arg}`);
        printHelp();
        process.exit(1);
    }
  }

  return args;
}

function printHelp() {
  console.log(`
Memory Consolidation CLI

Usage:
  memory-consolidation-cli [options]

Options:
  --input-dir <dir>       Directory containing daily log files (e.g., memory/)
  --input-file <file>     Single daily log file to consolidate
  --memory-file <file>    Output long-term memory file (default: MEMORY.md)
  --budget <number>       Events to retain per day (default: 10)
  --start-date <YYYY-MM-DD>  Start date for memory (default: first log date)
  --dry-run               Print consolidated memory without writing
  --verbose               Show detailed consolidation process
  --help                  Show this help message

Examples:
  # Consolidate all daily logs in memory/ directory
  memory-consolidation-cli --input-dir memory/ --memory-file MEMORY.md

  # Consolidate a single day
  memory-consolidation-cli --input-file memory/2026-05-21.md --memory-file MEMORY.md

  # Dry run (preview without writing)
  memory-consolidation-cli --input-dir memory/ --dry-run --verbose
`);
}

function loadMemory(filePath: string, startDate: Date): MemoryStore {
  if (existsSync(filePath)) {
    // TODO: Parse existing MEMORY.md to resume (for now, start fresh)
    console.log(`Loading existing memory from ${filePath}...`);
  }

  return new MemoryStore(startDate);
}

function getDailyLogFiles(dir: string): string[] {
  if (!existsSync(dir)) {
    throw new Error(`Directory not found: ${dir}`);
  }

  const files = readdirSync(dir)
    .filter(f => /\d{4}-\d{2}-\d{2}\.md/.test(f))
    .map(f => join(dir, f))
    .filter(f => statSync(f).isFile())
    .sort();

  return files;
}

function main() {
  const args = parseArgs();

  try {
    let dailyFiles: string[] = [];

    if (args.inputDir) {
      dailyFiles = getDailyLogFiles(args.inputDir);
      console.log(`Found ${dailyFiles.length} daily log files in ${args.inputDir}`);
    } else if (args.inputFile) {
      if (!existsSync(args.inputFile)) {
        throw new Error(`File not found: ${args.inputFile}`);
      }
      dailyFiles = [args.inputFile];
    } else {
      console.error('Error: Must specify --input-dir or --input-file');
      printHelp();
      process.exit(1);
    }

    if (dailyFiles.length === 0) {
      console.error('No daily log files found');
      process.exit(1);
    }

    // Determine start date
    const firstFile = dailyFiles[0];
    const match = firstFile.match(/(\d{4}-\d{2}-\d{2})/);
    const startDate = args.startDate
      ? new Date(args.startDate)
      : match
        ? new Date(match[1])
        : new Date();

    console.log(`Start date: ${startDate.toISOString().split('T')[0]}`);
    console.log(`Budget: ${args.budgetPerDay} events per day\n`);

    const memory = loadMemory(args.memoryFile, startDate);

    for (const file of dailyFiles) {
      const match = file.match(/(\d{4}-\d{2}-\d{2})/);
      if (!match) continue;

      const dateStr = match[1];
      const date = new Date(dateStr);
      const content = readFileSync(file, 'utf-8');
      const events = parseDailyLog(content, date);

      if (args.verbose) {
        console.log(`Processing ${dateStr}: ${events.length} events`);
      }

      consolidate(events, memory, args.budgetPerDay);

      if (args.verbose) {
        console.log(`  Memory size: ${memory.size()} events, ${memory.patterns.length} patterns`);
      }
    }

    console.log(`\nConsolidation complete!`);
    console.log(`  Total events in memory: ${memory.size()}`);
    console.log(`  Detected patterns: ${memory.patterns.length}`);
    console.log(`  Compression ratio: ${(dailyFiles.length * 20) / memory.size()}x\n`);

    const output = memory.toMarkdown();

    if (args.dryRun) {
      console.log('Dry run - output preview:\n');
      console.log(output.substring(0, 500) + '...\n');
    } else {
      writeFileSync(args.memoryFile, output, 'utf-8');
      console.log(`Long-term memory saved to ${args.memoryFile}`);
    }
  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

main();
