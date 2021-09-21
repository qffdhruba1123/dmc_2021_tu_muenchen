// Scraping Amazon for book recommendations - Part 1
// This script retrieves the Amazon URL for books in the test set.

// Activate puppeteer
const puppeteer = require('puppeteer');
const fs = require('fs').promises;

// This function submits relevant data from books in test set into Amazon Advanced Book Search and gets the URL of the first recommendation.
async function getAmazonURL(title, author, publisher, page, run) {
  // Open advanced book search
  await page.goto('https://www.amazon.de/advanced-search/books', {waitUntil: 'networkidle2'});
  await page.waitForSelector('input[name=field-author]', 'input[name=field-title]','input[name=field-isbn]','input[name=field-publisher]','select[name=field-language]','select[name=field-binding_browse-bin]' );
  // Accept Cookies
  if (run == 0) { 
    await page.click('input[name="accept"]');
  };
  // Select type: Taschenbuch
  await page.select('#asMain > tbody > tr:nth-child(2) > td:nth-child(2) > div:nth-child(2) > select', '492559011');
  // Select sort: salesrank
  await page.select('#asMain > tbody > tr:nth-child(2) > td:nth-child(2) > div:nth-child(5) > select', 'salesrank');
  // Enter title
  await page.$eval('input[name=field-title]', (el, title) => {el.value = title},title);
  // Enter author
  await page.$eval('input[name=field-author]', (el,author) => {el.value = author},author);
  // Enter publisher
  await page.$eval('input[name=field-publisher]', (el,publisher) => {el.value = publisher},publisher);
  // Submit search
  await page.click('input[name="Adv-Srch-Books-Submit"]');
  await page.waitForNavigation({ waitUntil: 'networkidle2' });

  // Get url of first recommendation
  try {
    const href = await page.evaluate(() => {return document.getElementsByClassName('a-link-normal a-text-normal').item(0).href});
    return href
  } catch (error) {
    // Throw an error if no recommendations exist
    const href = 'not found';
    return href
  }
};


// This function gets and iterates over books in test set.
async function main(start, end) {
    // Launch browser
    const browser = await puppeteer.launch({headless: false});
    const page = await browser.newPage();
    // Get data from test set
    let eval_merge = await fs.readFile('');
    let eval_merge_parsed = await JSON.parse(eval_merge);

    // Iterate over test set
    for(var i = start; i < end; i++) {
        var run = i - start;
        var item = await eval_merge_parsed[i];
        // Call getAmazonURL with relevant data
        const url = await getAmazonURL(item.title, item.author, item.publisher, page, run);
        // Mimic human behavior by waiting random time
        const secondToWait = (Math.floor(Math.random() * 5) + 1)*1000;
        await page.waitForTimeout(secondToWait);
        // Save url
        item['url']= url;
    };
    // Create new Json file
    await fs.writeFile('',JSON.stringify(eval_merge_parsed))
    // Close browser
    await browser.close();
};

// Run scraper for entire test set.
main(0,998);
