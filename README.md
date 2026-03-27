# Bot-B-Gone 🤥

**An open-source framework for filtering bot noise and finding the true human open rate of your email newsletter.**

> *"Your open rate is a lie. A structural distortion built into every major ESP. And almost nobody in this industry will talk about it plainly."* — [Read the full manifesto in A Media Operator](#)

## The Problem

More than half of all email opens are bots. Up to 97% of clicks are non-human. Security scanners, link prefetchers, and corporate mail proxies are firing pixels and following links before a person ever touches the email.

Every Email Service Provider (ESP) has a bot filter. But ESPs are incentivized to show you a number that feels good, not a number that is perfectly accurate. High open rates keep customers happy. 

If you are a publisher or platform selling advertising against those numbers, you are flying on instruments you didn't build, can't inspect, and don't control.

## The Solution

Bot-B-Gone is a shared industry framework for taking back control of your data. It provides:

1. **The Methodology:** How to use active honeypots and behavioral signals to find your true baseline.
2. **The Algorithm:** A foundational model for filtering raw event data (user agents, IPs, timing).
3. **The ESP Guide:** Exactly what raw data to demand from Sailthru, SendGrid, Mailchimp, beehiiv, and others.
4. **The Standard:** A "Confidence Scorecard" for publishers to transparently report their metrics to advertisers.

## Repository Structure

* `/docs/METHODOLOGY.md` - The core concepts: honeypots, the FP/FN tradeoff, and the CTOR math check.
* `/docs/MEASURING_SUCCESS.md` - The Confidence Scorecard and how to standardize reporting.
* `/docs/ESP_GUIDE.md` - How to extract the necessary raw event data from major ESPs.
* `/model/` - The bot-filtering algorithm (SQL/Python reference implementations).
* `/charts/` - Visualizations of the tradeoff curve and confidence decay.

## Why Open Source?

Display advertising went through this exact reckoning 15 years ago with impression fraud. The response was to create shared standards (like the IAB). Email needs the same thing. 

There is nothing proprietary about pattern matching on user agents, IP behavior, timing signatures, and click cadences. The only reason these models are black boxes is because transparency would expose how much noise each platform lets through.

## How to Contribute

If you are a publisher, send this to your data team. 
If you are an engineer, **please star the repo, fork it, and contribute.** Add the patterns you're seeing from your ESP. Help us build the shared standard this industry desperately needs.

---
*Bot-B-Gone was initiated to help publishers build trust with advertisers through radical data transparency.*

## The ESP Challenge

We don't want to fight ESPs. We want to align incentives. We are challenging every major ESP to run the Bot-B-Gone Framework against their own data and publish the results. 

[Read the full ESP Challenge here.](/docs/THE_ESP_CHALLENGE.md)
